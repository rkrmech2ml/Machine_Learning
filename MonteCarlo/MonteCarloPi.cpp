#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <fstream>



double MonteCarlo(int N, std::mt19937 &gen, std::uniform_real_distribution<> &dist) // random number generator header file is <random>
{
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

    // Different sample sizes
    std::vector<int> Ns;
    int start = 100, end = 1000000, steps = 100;
    double step_size = (end - start) / static_cast<double>(steps - 1);
    for (int i = 0; i < steps; ++i) {
        Ns.push_back(static_cast<int>(start + i * step_size));
    }

    // Open file to save results (optional)
    std::ofstream fout("pi_estimates.csv");
    fout << "N,Estimate,Error\n";

    for (int N : Ns) 
    {
        double pi_est = MonteCarlo(N, gen, dist);
        double error = fabs(pi_est - M_PI);

        std::cout << "N = " << N
                  << " | Estimate = " << pi_est
                  << " | Error = " << error << std::endl;

        fout << N << "," << pi_est << "," << error << "\n";
    }

    fout.close();
    return 0;
}
