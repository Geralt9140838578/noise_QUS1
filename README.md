# noise_QUS1 - QUS_1 :: Firstly, we generate three waves and subsequently merge them into a single wave. Next, we introduce noise to every point on this wave. Then, we utilize SVD technology to eliminate the noise. Finally, we plot the newly obtained noise-free wave on a graph.


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <fstream>

const double PI = 3.141592653589793238463;

void display(std::vector<double> buffer){
    for(auto x : buffer){
        std::cout<<x<<", ";
    }

    std::cout<<"\n";

}

std::vector<double> addNoise(std::vector<double> wave, double noise)
{
    std::vector<double> buffer(wave.size());

    std::default_random_engine rng(std::random_device{}());
    std::normal_distribution<double> dist(0, sqrt(pow(10, -noise/10)));
    for (int i = 0; i < wave.size(); i++)
    {
        buffer[i] += dist(rng);
    }

    return buffer;
}

std::vector<double> generateMixedSineWave(double frequency1, double amplitude1, double frequency2, double amplitude2, double frequency3, double amplitude3, double sampleRate, double duration)
{
    const int numSamples = (int)(sampleRate * duration);
    std::vector<double> buffer(numSamples);

    for (int i = 0; i < numSamples; i++)
    {
        double t = (double)i / sampleRate;
        buffer[i] = amplitude1 * sin(2 * PI * frequency1 * t)
                  + amplitude2 * sin(2 * PI * frequency2 * t)
                  + amplitude3 * sin(2 * PI * frequency3 * t);
    }
    

    return buffer;
}

Eigen::MatrixXd svdDenoise(const Eigen::MatrixXd& matrix, double tolerance)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singularValues = svd.singularValues();

    int numSingularValues = singularValues.size();
    int numSingularValuesToKeep = 0;
    for (int i = 0; i < numSingularValues; i++)
    {
        if (singularValues(i) > tolerance)
            numSingularValuesToKeep++;
    }

    Eigen::MatrixXd u = svd.matrixU().leftCols(numSingularValuesToKeep);
    Eigen::MatrixXd v = svd.matrixV().leftCols(numSingularValuesToKeep);
    Eigen::VectorXd s = singularValues.head(numSingularValuesToKeep);

    Eigen::MatrixXd denoisedMatrix = u * s.asDiagonal() * v.transpose();

    return denoisedMatrix;
}

void writeVectorToFile(const std::vector<double>& data, const std::string& filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); i++)
    {
        outfile << float(i)*10000 << "," << data[i] << std::endl;
    }

    outfile.close();
}

int main()
{
    const double sampleRate = 44100; // samples per second
    const double frequency1 = 440;   // Hz
    const double frequency2 = 880;   // Hz
    const double frequency3 = 1320;  // Hz
    const double amplitude1 = 0.5;
    const double amplitude2 = 0.3;
    const double amplitude3 = 0.2;
    const double duration = 5;       // seconds

    std::vector<double> wave = generateMixedSineWave(frequency1, amplitude1, frequency2, amplitude2, frequency3, amplitude3, sampleRate, duration);
    std::vector<double> noisyWave = addNoise(wave, -20);


    Eigen::Map<Eigen::VectorXd> matrix(noisyWave.data(), noisyWave.size());
    Eigen::MatrixXd denoisedMatrix = svdDenoise(matrix.transpose(), 0.05);

    // Convert the denoised matrix back to a vector
    std::vector<double> denoisedSineWave(denoisedMatrix.data(), denoisedMatrix.data() + denoisedMatrix.rows() * denoisedMatrix.cols());

    std::cout << "The last sample value of the original sine wave is: " << wave.back() << std::endl;
    writeVectorToFile(wave, "wave.csv");
    std::cout << "The last sample value of the noisy sine wave is: " << noisyWave.back() << std::endl;
    writeVectorToFile(noisyWave, "noisyWave.csv");

    std::cout << "The last sample value of the denoised sine wave is: " << denoisedSineWave.back() << std::endl;
    writeVectorToFile(denoisedSineWave, "denoisedSineWave.csv");


    return 0;
}
