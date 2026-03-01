#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using Clock = chrono::high_resolution_clock;

bool read_matrix(const string &filename, int &n, vector<double> &M) {
    ifstream in(filename);
    if (!in) {
        cerr << "Cant open file for reading:  " << filename << "\n";
        return false;
    }
    if (!(in >> n)) {
        cerr << "Error reading matrix size from file: " << filename << "\n";
        return false;
    }
    if (n <= 0) {
        cerr << "Incorrect matrix size: " << n << "\n";
        return false;
    }
    M.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(in >> M[(size_t)i * n + j])) {
                cerr << "Error reading element (" << i << "," << j << ") from " << filename << "\n";
                return false;
            }
        }
    }
    return true;
}

bool write_matrix(const string &filename, int n, const vector<double> &M) {
    ofstream out(filename);
    if (!out) {
        cerr << "Cant open file for writing: " << filename << "\n";
        return false;
    }
    out << n << "\n";
    out << fixed << setprecision(4);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << M[(size_t)i * n + j];
            if (j + 1 < n) out << " ";
        }
        out << "\n";
    }
    return true;
}
void multiply(int n, const vector<double> &A, const vector<double> &B, vector<double> &C) {
    fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A[(size_t)i * n + k];
            for (int j = 0; j < n; ++j) {
                C[(size_t)i * n + j] += aik * B[(size_t)k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " A.txt B.txt result.txt\n\n"
             << "Input file formatting: first line - n (whole number, matrix size), next n lines with n numbers each (double).\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileOut = argv[3];

    int nA = 0, nB = 0;
    vector<double> A, B, C;

    if (!read_matrix(fileA, nA, A)) return 2;
    if (!read_matrix(fileB, nB, B)) return 3;

    if (nA != nB) {
        cerr << "Matrices must be same size (n x n). nA=" << nA << ", nB=" << nB << "\n";
        return 4;
    }
    int n = nA;
    C.assign((size_t)n * n, 0.0);

    unsigned long long n64 = (unsigned long long)n;
    unsigned long long mults = n64 * n64 * n64;
    unsigned long long adds = mults > (unsigned long long)n64 * n64 ? (mults - n64 * n64) : 0;
    unsigned long long flops = mults + adds;
    size_t mem_bytes = 3 * (size_t)n * (size_t)n * sizeof(double);

    cout << "Matrix size n = " << n << "\n";
    cout << "Operation count: multiplications = " << mults
         << ", additions = " << adds << ", total FLOPs ~ " << flops << "\n";
    cout << "Memory (A,B,C) ~ " << mem_bytes << " bytes\n";

    auto t0 = Clock::now();

    multiply(n, A, B, C);

    auto t1 = Clock::now();
    chrono::duration<double> elapsed = t1 - t0;
    double secs = elapsed.count();
    long long millis = chrono::duration_cast<chrono::milliseconds>(elapsed).count();

    if (!write_matrix(fileOut, n, C)) {
        cerr << "Error writing the result matrix\n";
        return 5;
    }

    cout << fixed << setprecision(6);
    cout << "Multiplication time: " << secs << " sec (" << millis << " ms)\n";

    string statsFile = fileOut.erase(fileOut.length()-4) + "_stats.txt";
    ofstream st(statsFile);
    if (st) {
        st << "InputA: " << fileA << "\n";
        st << "InputB: " << fileB << "\n";
        st << "Output: " << fileOut << "\n";
        st << "n = " << n << "\n";
        st << "mults = " << mults << "\n";
        st << "adds = " << adds << "\n";
        st << "FLOPs ~ " << flops << "\n";
        st << "memory bytes (A,B,C) ~ " << mem_bytes << "\n";
        st << "time_sec = " << secs << "\n";
        st << "time_ms = " << millis << "\n";
        st.close();
        cout << "stats saved to " << statsFile << "\n";
    } else {
        cerr << "Error writing stats file: " << statsFile << "\n";
    }

    return 0;
}

