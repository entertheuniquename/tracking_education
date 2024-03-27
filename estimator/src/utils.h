#pragma once

#include <armadillo>
#include <complex>

namespace Utils {

template <typename Type>
arma::Col<Type> diag (const arma::Mat<Type>& x) {
    return x.diag();
}

inline double eps (const double& x = 1.0) {
    auto xp = std::abs(x);
    return std::nextafter(xp, xp + 1.0f) - xp;
}

inline bool isSymmetricPositiveSemiDefinite (const std::string& name, const arma::Mat<double>& x) {

    if (x.is_empty()) {
        throw std::invalid_argument("Matrix " + name + " is empty.");
    }

    if (!x.is_square()) {
        throw std::invalid_argument("Matrix " + name + " isn't square.");
    }

    auto tol = 100.*max(eps(abs(diag(x))));

    if (!x.is_symmetric()) {
        arma::mat d = abs(x - trans(x));
        if (!find(d>sqrt(tol)).is_empty()) {
            std::stringstream ss;
            ss << "sqrt(tol)=" << sqrt(tol) << std::endl;
            ss << "abs(x - trans(x))=" << std::endl << d << std::endl;
            ss << "Matrix: " << std::endl << x << std::endl << " isn't symmetric.";
            throw std::invalid_argument(ss.str());
        }
    }

    auto notPositiveSemidefinite = any(find(eig_sym((x + x.t())/2.) < -tol ));

    if (notPositiveSemidefinite) {
        throw std::invalid_argument("Matrix " + name + " isn't positive definite.");
    }

    return true;
}

inline arma::mat svdPSD(const arma::mat &A) {
    arma::mat U,V;
    arma::vec s;
    svd(U,s,V,A);
    return V*sqrt(diagmat(s));
}

inline arma::Mat<double>
qrFactor(const arma::Mat<double>& A,
         const arma::Mat<double>& S,
         const arma::Mat<double>& Ns
         ) {
    arma::Mat<double> D = join_cols(S.t()*A.t(),Ns.t()), Q, R;
    qr_econ(Q, R, D);
    return R.t();
}

inline arma::Mat<double>
cholPSD(const arma::Mat<double>& A) {
    arma::Mat<double> ret;
    if (!chol(ret, A)) {
        return svdPSD(A);
    }
    return ret.t();
}

inline size_t length(const arma::Mat<double>& m) {
    return m.n_rows * m.n_cols;
}

inline arma::Mat<double>
zeros(size_t n, size_t m) {
    return arma::zeros(n, m);
}

inline arma::Mat<double> ComputeKalmanGain(arma::Mat<double> Sy,
                                           arma::Mat<double> Pxy) {
    arma::Mat<double> K1 = arma::solve(arma::trimatl(Sy), trans(Pxy), arma::solve_opts::fast);
    arma::Mat<double> K  = arma::trans(arma::solve(arma::trimatu(trans(Sy)), K1, arma::solve_opts::fast));
    return K;
}


inline double ComputeAngleDifference(double a1, double a2) {
    return std::arg(std::complex<double>(cos(a1 - a2), sin(a1 - a2)));
}

}

#define PRINTM(x) //std::cerr << #x << std::endl << x << __FILE__ << ":" << __LINE__ << std::endl << std::endl
#define CHECK_SYMETRIC_POSITIVE(x) Utils::isSymmetricPositiveSemiDefinite(#x, x)

