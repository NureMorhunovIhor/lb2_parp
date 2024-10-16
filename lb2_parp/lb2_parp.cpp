#include <iostream>
#include <cmath>         
#include <immintrin.h>  
#include <chrono>       
#include <random>        
#include <complex>

const size_t ARRAY_SIZE = 4096 * 4096;

float sum_sqrt_float(const float* y, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += std::sqrt(y[i]);  // x[i] = sqrt(y[i])
    }
    return sum;
}

float sum_sqrt_float_simd(const float* y, size_t n) {
    float sum = 0.0f;

    for (size_t i = 0; i < n; i += 8) {
        __m256 data = _mm256_loadu_ps(&y[i]);
        __m256 sqrt_data = _mm256_sqrt_ps(data);

        float result[8];
        _mm256_storeu_ps(result, sqrt_data);

        for (int j = 0; j < 8; ++j) {
            sum += result[j];
        }
    }

    return sum;
}


double sum_sqrt_double(const double* y, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::sqrt(y[i]);
    }
    return sum;
}

double sum_sqrt_double_simd(const double* y, size_t n) {
    __m256d simd_sum = _mm256_setzero_pd();

    for (size_t i = 0; i < n; i += 4) {
        __m256d data = _mm256_loadu_pd(&y[i]);
        __m256d sqrt_data = _mm256_sqrt_pd(data);
        simd_sum = _mm256_add_pd(simd_sum, sqrt_data);
    }

    double result[4];
    _mm256_storeu_pd(result, simd_sum);

    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += result[i];
    }
    return sum;
}

template <typename T>
void generate_random_array(T* arr, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(1.0, 100.0);

    for (size_t i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }
}

template <typename Func>
void measure_execution_time(const std::string& name, Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took " << duration.count() << " seconds.\n";
}

void task6() {
    float* float_data = new float[ARRAY_SIZE];
    double* double_data = new double[ARRAY_SIZE];

    generate_random_array(float_data, ARRAY_SIZE);
    generate_random_array(double_data, ARRAY_SIZE);

    measure_execution_time("Regular sum of square roots (float)", [&]() {
        float sum = sum_sqrt_float(float_data, ARRAY_SIZE);
        std::cout << "Sum (regular float): " << sum << std::endl;
        });

    measure_execution_time("SIMD sum of square roots (float)", [&]() {
        float simd_sum = sum_sqrt_float_simd(float_data, ARRAY_SIZE);
        std::cout << "Sum (SIMD float): " << simd_sum << std::endl;
        });

    measure_execution_time("Regular sum of square roots (double)", [&]() {
        double sum = sum_sqrt_double(double_data, ARRAY_SIZE);
        std::cout << "Sum (regular double): " << sum << std::endl;
        });

    measure_execution_time("SIMD sum of square roots (double)", [&]() {
        double simd_sum = sum_sqrt_double_simd(double_data, ARRAY_SIZE);
        std::cout << "Sum (SIMD double): " << simd_sum << std::endl;
        });

    delete[] float_data;
    delete[] double_data;
}

void multiply_complex_no_simd(const std::complex<float>* y,
    const std::complex<float>* z,
    std::complex<float>* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = y[i] * z[i];  // Стандартне комплексне множення
    }
}

// Функція з використанням SIMD (AVX)
void multiply_complex_simd(const float* y, const float* z, float* x, size_t n) {
    for (size_t i = 0; i < n * 2; i += 8) {  // Обробка 8 float одночасно
        __m256 y_vals = _mm256_loadu_ps(&y[i]);
        __m256 z_vals = _mm256_loadu_ps(&z[i]);
        __m256 result = _mm256_mul_ps(y_vals, z_vals);  // SIMD-множення
        _mm256_storeu_ps(&x[i], result);
    }
}

template <typename T>
void generate_random_complex(T* arr, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);

    for (size_t i = 0; i < n; ++i) {
        arr[i] = { dist(gen), dist(gen) };
    }
}

void task8() {
    auto n = ARRAY_SIZE / 2;  // Кількість комплексних чисел

    // Виділення пам'яті для масивів
    std::complex<float>* y = new std::complex<float>[n];
    std::complex<float>* z = new std::complex<float>[n];
    std::complex<float>* x_no_simd = new std::complex<float>[n];
    float* y_flat = new float[n * 2];  // Плаский масив для SIMD
    float* z_flat = new float[n * 2];
    float* x_simd = new float[n * 2];

    generate_random_complex(y, n);
    generate_random_complex(z, n);

    auto start = std::chrono::high_resolution_clock::now();
    multiply_complex_no_simd(y, z, x_no_simd, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Without SIMD: " << duration.count() << " seconds\n";

    // Конвертація комплексних чисел у пласкі масиви
    for (size_t i = 0; i < n; ++i) {
        y_flat[2 * i] = y[i].real();
        y_flat[2 * i + 1] = y[i].imag();
        z_flat[2 * i] = z[i].real();
        z_flat[2 * i + 1] = z[i].imag();
    }

    start = std::chrono::high_resolution_clock::now();
    multiply_complex_simd(y_flat, z_flat, x_simd, n);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "SIMD: " << duration.count() << " seconds\n";

    delete[] y;
    delete[] z;
    delete[] x_no_simd;
    delete[] y_flat;
    delete[] z_flat;
    delete[] x_simd;
}
void shift_256bit_simd(const __m256i* data, __m256i* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = _mm256_slli_epi64(data[i], 1);  // Сдвиг на 1 бит
    }
}

void task9_avx2() {
    const size_t n = ARRAY_SIZE / 4;

    std::vector<__m256i> data(n, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF));
    std::vector<__m256i> result(n);

    auto start = std::chrono::high_resolution_clock::now();
    shift_256bit_simd(data.data(), result.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Shifting 256-bit blocks by 1 bit took: " << duration.count() << " seconds\n";
}


// ||||||||||||||| Провевить на своих ноутах и компах!!!! ||||||||||||||||||||||||||
void shift_512bit_simd(const __m512i* data, __m512i* result, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        result[i] = _mm512_slli_epi64(data[i], 1);
    }
}

void task9() {
    const size_t n = ARRAY_SIZE / 8;

    std::vector<__m512i> data(n, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
    std::vector<__m512i> result(n);

    auto start = std::chrono::high_resolution_clock::now();
    shift_512bit_simd(data.data(), result.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Shifting 512-bit blocks by 1 bit took: " << duration.count() << " seconds\n";
}
// ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||



int main() {
    //task6();
    //task8();
    task9_avx2();
    return 0;
}
