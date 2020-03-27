/*
 * Copyright (C) 2019 Dillon Cislo
 *
 * This file is part of PlaQuaGE.
 *
 * PlaQuaGE is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 */

#ifndef _GENERALIZED_LAPLACIAN_H_
#define _GENERALIZED_LAPLACIAN_H_

#include <complex>
#include <vector>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace PlaQuaGE {

  ///
  /// A function class that constructs the sparse matrix representation of the Beltrami
  /// Equation.  Details can be found in Lam and Lui (2014).  Mainly for internal use.
  ///
  /// Templates:
  ///   DerivedV    Derived data type of Eigen matrix for V (e.g. double from MatrixXd)
  ///   DerivedF    Derived int type of Eigen matrix for F (e.g. int from MatrixXi)
  ///
  /// Inputs:
  ///   V     #V by 2 list of mesh vertex positions
  ///   F     #F by 3 list of mesh faces
  ///   mu    #F by 1 list of the complex Beltrami coefficient defined on faces
  ///
  /// Outputs:
  ///   A     #V by #V sparse matrix Beltrami operator
  ///
  template<typename DerivedV, typename DerivedF>
  class Generalized_Laplacian {

    public:

      static Eigen::SparseMatrix<DerivedV> Build(
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F,
          const Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mu ) {

        // Check that the Beltrami coefficient vector is properly sized
        if ( mu.size() != F.rows() ) {
          std::invalid_argument("Beltrami coefficient vector is improperly sized.");
        }

        // Construct the coefficient matrices for each face
        std::vector< Eigen::Matrix<DerivedV, 2, 2> > M;
        M.reserve( F.rows() );
        for( int f = 0; f < F.rows(); f++ ) {

          DerivedV reMu = mu(f).real();
          DerivedV imMu = mu(f).imag();
          DerivedV absMu = std::abs( mu(f) );

          DerivedV af, bf, gf;

          af = ( 1.0 - 2.0 * reMu + absMu * absMu ) / ( 1.0 - absMu * absMu );
          bf = -2.0 * imMu / ( 1.0 - absMu * absMu );
          gf = ( 1.0 + 2.0 * reMu + absMu * absMu ) / ( 1.0 - absMu * absMu );

          Eigen::Matrix<DerivedV, 2, 2> Mf;
          Mf << gf, -bf, -bf, af;

          M.push_back( Mf );

        }

        // Extract facet edges (as row vectors) and unsigned facet (double) areas
        // NOTE: This expects that the facets are oriented CCW
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> doubleArea( F.rows(), 1 );
        std::vector< std::vector< Eigen::Matrix<DerivedV, 1, 2> > > e;
        e.reserve( F.rows() );
        for( int f = 0; f < F.rows(); f++ ) {

          Eigen::Matrix<DerivedV, 1, 2> e0, e1, e2;
          e0 = V.row( F(f,2) ) - V.row( F(f,1) );
          e1 = V.row( F(f,0) ) - V.row( F(f,2) );
          e2 = V.row( F(f,1) ) - V.row( F(f,0) );

          std::vector< Eigen::Matrix<DerivedV, 1, 2> > ef{ e0, e1, e2 };

          e.push_back( ef );

          doubleArea(f) = std::abs( e0(0) * e1(1) - e1(0) * e0(1) );

        }

        // Construct sparse matrix triplets
        typedef Eigen::Triplet<DerivedV> T;
        std::vector<T> tList;
        tList.reserve( 9 * F.rows() );
        for( int f = 0; f < F.rows(); f++ ) {
          for( int i = 0; i < 3; i++ ) {
            for( int j = 0; j < 3; j++ ) {

              DerivedV vij;
              vij = (DerivedV) ( e[f][i] * M[f] * e[f][j].transpose() );
              vij = -vij / doubleArea(f);
              // vij = -e[f][i] * M[f] * e[f][j].transpose() / doubleArea(f);

              tList.push_back( T( F(f,i), F(f,j), vij ) );

            }
          }
        }

        // Complete sparse matrix construction
        Eigen::SparseMatrix<DerivedV> A( V.rows(), V.rows() );
        A.setFromTriplets( tList.begin(), tList.end() );

        return A;

      };

  };

} // namespace PlaQuaGE

#endif // _GENERALIZED_LAPLACIAN_H_
