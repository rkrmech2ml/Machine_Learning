#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>

class MarkovChainPi {
private:
    std::mt19937 gen;
    std::normal_distribution<> step_dist;
    
public:
    MarkovChainPi(unsigned seed = std::random_device{}()) 
        : gen(seed), step_dist(0.0, 0.1) {}  // Small steps for random walk
    
    // Markov chain random walk to estimate pi
    double estimate_pi_markov(int N, double step_size) {
        // Start at a random point in the square [-1,1] x [-1,1]
        std::uniform_real_distribution<> init_dist(-1.0, 1.0);
        double x = init_dist(gen);
        double y = init_dist(gen);
        
        // Update step distribution
        step_dist = std::normal_distribution<>(0.0, step_size);
        
        int inside_circle = 0;
        int valid_samples = 0;
        
        for (int i = 0; i < N; ++i) {
            // Propose new position (Markov chain step)
            double new_x = x + step_dist(gen);
            double new_y = y + step_dist(gen);
            
            // Keep within bounds [-1,1] x [-1,1] (reflecting boundary)
            if (new_x < -1.0) new_x = -1.0 - (new_x + 1.0);
            if (new_x > 1.0) new_x = 1.0 - (new_x - 1.0);
            if (new_y < -1.0) new_y = -1.0 - (new_y + 1.0);
            if (new_y > 1.0) new_y = 1.0 - (new_y - 1.0);
            
            // Update current position
            x = new_x;
            y = new_y;
            
            // Check if current position is inside unit circle
            if (x * x + y * y <= 1.0) {
                inside_circle++;
            }
            valid_samples++;
        }
        
        return 4.0 * inside_circle / valid_samples;
    }
};
double MonteCarlo(int N, std::mt19937 &gen, std::uniform_real_distribution<> &dist) {
    int inside_circle = 0;
    for (int i = 0; i < N; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) {
            inside_circle++;
        }
    }
    return 4.0 * inside_circle / N;
}

int main() {
    // Random number setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    
    MarkovChainPi mc_pi;
    
    // Test different sample sizes
    std::vector<int> Ns = {50000};
    std::vector<double> step_sizes = {0.1};
    
    std::cout << "=== Comparing Methods ===" << std::endl;
    std::cout << "True π = " << M_PI << std::endl << std::endl;
    
    
    
    // Save comparison across different sample sizes
    std::ofstream fout("pi_comparison.csv");
    fout << "Method,N,Estimate,Error\n";
    
    // Test different sample sizes like your original code
    std::vector<int> test_Ns;
    int start = 100, end = 1000000, steps = 100;
    double step_size = (end - start) / static_cast<double>(steps - 1);
    for (int i = 0; i < steps; ++i) {
        test_Ns.push_back(static_cast<int>(start + i * step_size));
    }
    
    std::cout << "Testing " << steps << " different sample sizes..." << std::endl;
    
    for (int N : test_Ns) {
        std::cout << "Testing N = " << N << "..." << std::endl;
        
        // Monte Carlo estimate
        double pi_mc = MonteCarlo(N, gen, dist);
        double error_mc = fabs(pi_mc - M_PI);
        fout << "MonteCarlo," << N << "," << pi_mc << "," << error_mc << "\n";
        
        // Markov Chain estimate
        double pi_markov = mc_pi.estimate_pi_markov(N, 0.1);
        double error_markov = fabs(pi_markov - M_PI);
        fout << "Markov," << N << "," << pi_markov << "," << error_markov << "\n";
        
        std::cout << "  MC: " << pi_mc << " (error: " << error_mc << ")" << std::endl;
        std::cout << "  Markov: " << pi_markov << " (error: " << error_markov << ")" << std::endl;
    }
    
    fout.close();
    std::cout << "pi_comparison.csv" << std::endl;
    
    return 0;
}