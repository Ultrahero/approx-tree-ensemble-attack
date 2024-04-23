#include <string.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <list>

#include <boost/filesystem.hpp>
#include "nlohmann/json.hpp"

#include "decision_forest.h"
#include "neighbor_attack.h"
#include "timing.h"
#include "utility.h"

using nlohmann::json;

using namespace cz;

namespace cz {

namespace {

std::map<int, std::vector<Point>> LoadMilpAdv(
    const std::string& milp_adv_path) {
  std::ifstream fin(milp_adv_path);
  json milp_adv_dict;
  fin >> milp_adv_dict;
  // assert(milp_adv_dict.is_object());

  std::map<int, std::vector<Point>> milp_adv;

  for (const auto& iter : milp_adv_dict.items()) {
    for (const auto& p_obj : iter.value()) {
      Point p(p_obj.size());

      for (int i = 0; i < p_obj.size(); ++i)
        p[i] = p_obj[i];

      milp_adv[std::stoi(iter.key())].emplace_back(std::move(p));
    }
  }

  return std::move(milp_adv);
}

struct AdvExampleReturn{
  double runtime;
  std::map<int,double> norm_sums;
  int actual_num_examples;
  std::map<int,double>* best_norms;
  std::vector<int> best_hamming_dists;
  std::vector<int> best_neighbor_dists;
};

/*
 * Generates Adversarial examples for the given config and writes them into the given vector adv_examples
 * Returns some stats on the process
*/
AdvExampleReturn GenerateAdvExamples(const Config& config){
  using namespace std::chrono;
  srand(0);

  if(config.verbosity) cout << "Benchmarking model_path:" << config.model_path
       << " inputs_path:" << config.inputs_path << endl;

  if(config.verbosity) cout << "Loading model..." << endl;
  auto attack = std::make_unique<NeighborAttack>(config);
  attack->LoadForestFromJson(config.model_path);

  if(config.verbosity) cout << "Loading inputs..." << endl;
  auto parsed_data = cz::LoadSVMFile(config.inputs_path.c_str(),
                                     config.num_features, config.feature_start);

  bool verify_hamming = !config.milp_adv.empty();
  std::map<int, std::vector<Point>> milp_adv;
  std::vector<int> best_hamming_dists;
  std::vector<int> best_neighbor_dists;
  if (verify_hamming) {
    milp_adv = LoadMilpAdv(config.milp_adv);
    if(config.verbosity){
      cout << "Got milp advs: " << milp_adv.size() << endl;
      cout << " Adv size: " << milp_adv[0][0].Size() << endl;
    }
  }

  bool log_adv_training_examples = (!config.adv_training_path.empty() or !config.outputs_path.empty()) and config.save_adv_examples;

  Timing::Instance()->StartTimer("Total Time");
  auto start_timer = high_resolution_clock::now();

  std::map<int, double> norm_sums;
  for (auto np : NeighborAttack::kAllowedNormTypes)
    norm_sums[np] = 0;

  int actual_num_example = 0;
  std::map<int,double> best_dist;
  int max_row =
      std::min((int)parsed_data.size(), config.offset + config.num_point);
  std::map<int, double> best_norms[max_row - config.offset+1];
  std::vector<std::pair<int, Point>> adv_examples;
  for (int row = config.offset; row < max_row; ++row) {
    int i = row - config.offset + 1;

    const auto& data = parsed_data[row];
    
    int y_pred = attack->PredictLabel(data.second);
    if(config.verbosity){
      cout << "Running testing example at line " << i << endl;
      cout << "Checking if the point is correctly classified..." << endl;
      cout << "Correct label:" << data.first << " Predict Label:" << y_pred
          << endl;
      if (data.first != y_pred) {
        cout << "Mis-classified point, skipping...";
        adv_examples.push_back(std::move(std::make_pair(-1, data.second))); //save the example but with -1 label to know it was already misclassified
        best_norms[i] = std::move(std::map<int, double>({{-1,0.0}, {1,0.0}, {2,0.0}})); //set norms to 0 as already misclassified
        continue;
      }
      cout << "Correctly classified point, attacking...";
      cout << "Progress " << i << "/" << config.num_point
          << endl;
    }else{
      if(data.first != y_pred) {
        if(log_adv_training_examples) adv_examples.push_back(std::move(std::make_pair(-1, data.second))); //save the example but with -1 label to know it was already misclassified
        best_norms[i] = std::move(std::map<int, double>({{-1,0.0}, {1,0.0}, {2,0.0}})); //set norms to 0 as already misclassified
        continue;
      };
    }

    auto result = attack->FindAdversarialPoint(data.second);
    bool is_success = result.success();
    best_norms[i] = result.best_norms;

    if (!result.success()) {
      printf("!!!Failed on example %d\n", i);
      continue;
    }
    ++actual_num_example;

    std::map<int, int> adv_labels;
    adv_labels[1] = attack->PredictLabel(result.best_points[1]);
    adv_labels[2] = attack->PredictLabel(result.best_points[2]);
    adv_labels[-1] = attack->PredictLabel(result.best_points[-1]);

    for (const auto& iter : adv_labels) {
      assert(iter.second != data.first);
    }

    for (auto np : NeighborAttack::kAllowedNormTypes)
      norm_sums[np] += result.best_norms[np];

    if (verify_hamming) {
      int index = i - 1;

      // Note: Actually we may not have the corresponding |milp_adv[index]|
      // since milp is doing the filtering based on |.model| and we are using |.json|.
      assert(milp_adv.find(index) != milp_adv.end());

      double best_milp_adv_norm = DBL_MAX;
      Point best_milp_adv;
      // Multi class MILP will produce multiple adv points.
      for (const auto& p : milp_adv[index]) {
        double norm = p.Norm(data.second, config.norm_type);
        if (norm < best_milp_adv_norm) {
          best_milp_adv_norm = norm;
          best_milp_adv = p;
        }
      }

      int best_milp_adv_label = attack->PredictLabel(best_milp_adv);
      assert(best_milp_adv_label != y_pred);

      int best_hamming_dist = INT_MAX;
      int best_neighbor_dist = INT_MAX;
      for (const auto& p : result.hist_points) {
        int hamming_dist = attack->HammingDistanceBetween(
            p, best_milp_adv, best_milp_adv_label, y_pred);
        best_hamming_dist = std::min(best_hamming_dist, hamming_dist);
        int neighbor_dist = attack->NeighborDistanceBetween(
            p, best_milp_adv, best_milp_adv_label, y_pred, data.second);
        best_neighbor_dist = std::min(best_neighbor_dist, neighbor_dist);
      }
      best_hamming_dists.push_back(best_hamming_dist);
      best_neighbor_dists.push_back(best_neighbor_dist);
    }
    //write to output if we have nonzero verbosity 
    if(config.verbosity){
      printf("===== Attack result for example %d/%d Norm(%d)=%lf =====\n", i, config.num_point, config.norm_type, result.best_norms[config.norm_type]);
      cout << "All Best Norms: " << result.ToNormString() << endl;

      cout << "Average Norms: ";
      for (auto np : NeighborAttack::kAllowedNormTypes)
        printf("Norm(%d)=%lf ", np, norm_sums[np] / actual_num_example);
      cout << endl;
    }


    if (log_adv_training_examples) {
      if(config.save_all){
        for (auto p : result.hist_points) {
          adv_examples.push_back(std::move(std::make_pair(data.first, p))); //save all points from all threads
        }
      }else{
        adv_examples.push_back(std::move(std::make_pair(data.first, result.best_points[config.norm_type]))); //save only the best example for our norm
      }
    }
  }

  auto end_timer = high_resolution_clock::now();
  double total_seconds =
      duration_cast<duration<double>>(end_timer - start_timer).count();
  Timing::Instance()->EndTimer("Total Time");

  if (log_adv_training_examples) {
    printf("\n\nWriting %ld generated adversarial examples to %s\n", adv_examples.size(), config.outputs_path.c_str() );
    FILE* fp;
    fp = fopen(config.outputs_path.c_str(), "w+");
    for (auto p : adv_examples) {
      fprintf(fp, "%d %s\n", p.first, p.second.ToDebugString().c_str());
    }
    fclose(fp);
  }
  // *best_norms = std::move(*best_norms);
  return AdvExampleReturn{
    .runtime = total_seconds,
    .norm_sums = norm_sums,
    .actual_num_examples = actual_num_example,
    .best_norms = best_norms,
    .best_hamming_dists = best_hamming_dists,
    .best_neighbor_dists = best_neighbor_dists
  };
}

void BenchmarkDistortion(const Config& config) {
  using namespace std::chrono;
  srand(0);

  if(config.verbosity){//disable if not needed, takes some time to measure these metrics
    Timing::Instance()->SetCollectHistogram(config.collect_histogram);
  }
  

  if(!config.verbosity){
    
  }
  auto generate_output = GenerateAdvExamples(config);
  auto total_seconds = generate_output.runtime;
  auto best_hamming_dists = generate_output.best_hamming_dists;
  auto best_neighbor_dists = generate_output.best_neighbor_dists;
  auto norm_sums = generate_output.norm_sums;
  auto actual_num_example = generate_output.actual_num_examples;

  bool verify_hamming = !config.milp_adv.empty();

  if (verify_hamming && config.verbosity) {
    printf("Best Hamming Distance (max: %d, median: %d, mean: %.2lf): %s\n",
           Max(best_hamming_dists), Median(best_hamming_dists),
           Mean(best_hamming_dists), ToDebugString(best_hamming_dists).c_str());
    printf("Best Neighbor Distance (max: %d, median: %d, mean: %.2lf): %s\n",
           Max(best_neighbor_dists), Median(best_neighbor_dists),
           Mean(best_neighbor_dists),
           ToDebugString(best_neighbor_dists).c_str());
  }
  if(config.verbosity){
    cout << "==============================" << endl;
    cout << "==============================" << endl;
    cout << "==============================" << endl;
    cout << "Results for config:" << config.config_path << endl;
    cout << "Average Norms: ";
    for (auto np : NeighborAttack::kAllowedNormTypes)
      printf("Norm(%d)=%lf ", np, norm_sums[np] / actual_num_example);
    cout << endl;
    cout << "--- Timing Metrics ---" << endl;
    cout << Timing::Instance()->CollectMetricsString();

    cout << "## Actual Examples Tested:" << actual_num_example << endl;
    cout << "## "
        << "Time per point: " << total_seconds / actual_num_example << endl;
  }
};

struct ModelStats {
  double test_accuracy;
  int num_test_examples;
  int num_trees;
};

ModelStats CalculateAccuracy(const Config& config) {
  auto attack = std::make_unique<NeighborAttack>(config);
  attack->LoadForestFromJson(config.model_path);
  auto parsed_data = cz::LoadSVMFile(config.inputs_path.c_str(),
                                     config.num_features, config.feature_start);

  int num_total = parsed_data.size();
  int num_correct = 0;
  int i = 1;
  for (const auto& data : parsed_data) {
    int y_pred = attack->PredictLabel(data.second);
    if (y_pred == data.first) {
      ++num_correct;
    } else {
      // cout << "Incorrect point at line:" << i << " y_pred:" << y_pred
      //      << " y_expected:" << data.first << endl;
      // cout << ToDebugString(
      //             attack->ForestForTesting()->ComputeScores(data.second))
      //      << endl;
    }
    ++i;
  }

  return ModelStats{(double)num_correct / num_total, num_total,
                    attack->ForestForTesting()->NumTreesForTesting()};
}

void VerifyModelAccuracy() {
  namespace fs = boost::filesystem;
  std::unordered_set<std::string> verified_models;
  std::vector<std::string> sorted_configs;
  for (const auto& p : fs::recursive_directory_iterator("configs")) {
    if (p.path().string().find(".json") != std::string::npos)
      sorted_configs.push_back(p.path().string());
  }
  sort(sorted_configs.begin(), sorted_configs.end());

  for (const auto& config_path : sorted_configs) {
    Config config(config_path.c_str());

    if (verified_models.find(config.model_path) != verified_models.end())
      continue;
    verified_models.insert(config.model_path);

    auto model_stats = CalculateAccuracy(config);
    auto model = config.model_path;
    printf(
        "Model: %-40s Classes: %d \t Accuracy: %.2f%% \t Points: %d \t Trees: "
        "%d\n\n",
        model.c_str(), config.num_classes, model_stats.test_accuracy * 100,
        model_stats.num_test_examples, model_stats.num_trees);
  }
}
/*
 * Calculates the attack distance required for each sample to be perturbed.
 * Data, Model, config are set in config,
 * order is a normal parameter: may be -1, 1, 2 (-1 is inf)
 * 
 * If order is not in given range, returns an empty vector.
*/
std::vector<double> AttackDistances(const Config& config){
  //disturb all points until misclassification and calculate accuracy with those in radius epsilon
  std::vector<double> distances;
  const int order = config.norm_type;

  if(std::find(NeighborAttack::kAllowedNormTypes.begin(), NeighborAttack::kAllowedNormTypes.end(), order) == NeighborAttack::kAllowedNormTypes.end()) return distances;//return empty if order is not allowed

  //now perturb all the data, similarly to benchmark distortion
  auto generate_out = GenerateAdvExamples(config);

  for(int i=0; i<config.num_point - config.offset;i++){
    distances.push_back(generate_out.best_norms[i][order]);
  }

  return distances;
}

void SaveAdvExamples(Config& config){
  config.save_adv_examples = true;
  auto ret = GenerateAdvExamples(config);
  // cout << std::to_string(ret.actual_num_examples) << endl;
  //write adv_examples as svm to path.
  //WriteSVMFile(config.outputs_path, &adv_examples); saving is handled in Generate
}

}  // namespace

}  // namespace cz

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Usage: ./lt_attack configs/breast_cancer_robust_20x500_norm2_lt-attack.json"
         << endl;
    cout << "\tOR" <<endl;
    // cout << "./lt_attack verify <config> \t - verifies the accuracy on the given samples" <<endl;
    cout << "./lt_attack distance <config> \t - computes the minimum attack distance per sample" <<endl;
    cout << "./lt_attack examples <config> \t - generates adv examples" <<endl;
    return 0;
  }
  //cout << "Using config:" << argv[1] << endl;

  Config config;
  bool debug = false;
  if(argc == 2) config = Config(argv[1]);
  else if (argc >= 3) config = Config(argv[2]);

  if(config.verbosity or debug){
    cout << "Config looks like:"<<endl;
    cout << "\t num_points: "<< std::to_string(config.num_point).c_str() << endl;
    cout << "\t feature count: " << std::to_string(config.num_features).c_str() << endl; 
    cout << "\t outputspath: " << config.outputs_path.c_str() << endl;
    cout << "\t model: " << config.model_path.c_str() << endl;
  }
    
  // if (strcmp(argv[1], "verify") == 0) {
  //   cout << "Verifing model accuracy..." << endl;
  //   VerifyModelAccuracy();
  //   return 0;
  // }
  if(argc>=3){
    if (strcmp(argv[1], "distance") == 0){
      //calculate distance
      auto distances = AttackDistances(config);
      if(config.verbosity) cout << "Generated adv examples with distances, count is:" << std::to_string(distances.size()) << ". Writing the distances to "<< config.outputs_path.c_str() << endl;
      //write the result to config.outputs_path
      std::ofstream fout(config.outputs_path);
      if(!fout.is_open()){
        cout << "ERROR: Tried writing to file " << config.outputs_path << " but file could not be opened." << endl;
        return 2;
      }
      for(auto d : distances){
        fout << std::to_string(d) + "\n";
      }

      fout.close();
      
      return 0;
    }
    else if (strcmp(argv[1], "examples") == 0){
      //generates adv examples
      SaveAdvExamples(config); //saves to path specified in config
      return 0;
    }
    else {
      cout << "Unknown command `" << argv[1] <<"`. Will exit here."<< endl;
      cout << "Usage: ./lt_attack configs/breast_cancer_robust_20x500_norm2_lt-attack.json"
         << endl;
      cout << "\tOR" <<endl;
      // cout << "./lt_attack verify <config> \t - verifies the accuracy on the given samples" <<endl;
      cout << "./lt_attack distance <config> \t - computes the minimum attack distance per sample" <<endl;
      cout << "./lt_attack examples <config> \t - generates adv examples" <<endl;
      return 1;
    }
  }else{
    BenchmarkDistortion(config);
    return 0;
  }
}
