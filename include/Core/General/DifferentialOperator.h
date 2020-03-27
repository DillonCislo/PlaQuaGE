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

#ifndef _DIFFERENTIAL_OPERATOR_H_
#define _DIFFERENTIAL_OPERATOR_H_

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <igl/cotmatrix.h>
#include <igl/LinSpaced.h>

namespace PlaQuaGE {

  ///
  /// A differential operator for computing the derivatives of functions defined
  /// on the vertices of a planar mesh triangulation.
  ///
  /// Templates:
  ///   DerivedV    Derived data type of Eigen matrix for V (e.g. double from MatrixXd)
  ///   DerivedF    Derived data type of Eigen matrix for F (e.g. int from MatrixXi)
  ///
  template <typename DerivedV, typename DerivedF>
  class DifferentialOperator {

    private:

      Eigen::SparseMatrix<DerivedV> m_Dx; // Derivative w.r.t. x
      Eigen::SparseMatrix<DerivedV> m_Dy; // Derivative w.r.t. y
      Eigen::SparseMatrix< std::complex<DerivedV> > m_Dz; // Derivative w.r.t. z = x + i y
      Eigen::SparseMatrix< std::complex<DerivedV> > m_Dc; // Derivative w.r.t. c = x - i y

      // The cotangent representation of the Laplace-Beltrami operator
      Eigen::SparseMatrix<DerivedV> m_L;

    public:

      ///
      /// Null constructor
      ///
      DifferentialOperator() {};

      ///
      /// Basic constructor
      ///
      /// Inputs:
      ///   V   #V by 2 list of mesh vertex positions
      ///   F   #F by 3 list of mesh faces
      ///
      DifferentialOperator( const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F ) {

        // Construct Laplace-Beltrami operator
        igl::cotmatrix( V, F, this->m_L );

        // Construct remaining derivate operators
        Construct_Derivatives( V, F );

      };

      ///
      /// Get the x-derivative operator
      ///
      Eigen::SparseMatrix<DerivedV> Dx() const { return m_Dx; };

      ///
      /// Get the y-derivative operator
      ///
      Eigen::SparseMatrix<DerivedV> Dy() const { return m_Dy; };

      ///
      /// Get the z-derivative operator
      ///
      Eigen::SparseMatrix< std::complex<DerivedV> > Dz() const { return m_Dz; };

      ///
      /// Get the c-derivative operator
      ///
      Eigen::SparseMatrix< std::complex<DerivedV> > Dc() const { return m_Dc; };

      ///
      /// Get the Laplace-Beltrami operator
      ///
      Eigen::SparseMatrix<DerivedV> Laplacian() const { return m_L; };

    private:

      ///
      /// Constructs the basic derivative operators for a given planar mesh
      /// Each operator is a  #F by #V sparse matrix that acts on functions
      /// defined on vertices to return derivatives defined on mesh faces
      ///
      void Construct_Derivatives( const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F ) {

        // The number of vertices
        int nv = V.rows();

        // The number of mesh faces
        int nf = F.rows();

        Eigen::Matrix<DerivedF, Eigen::Dynamic, 1> u;
        u = igl::LinSpaced<Eigen::Matrix<DerivedF, Eigen::Dynamic, 1 > >( nf, 0, (nf-1) );

        // Extract facet edges
        // NOTE: This expects that the facets are oriented CCW
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> e1(nf, 2);
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> e2(nf, 2);
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> e3(nf, 2);
        for( int i = 0; i < nf; i++ ) {

          e1.row(i) = V.row( F(i,2) ) - V.row( F(i,1) );
          e2.row(i) = V.row( F(i,0) ) - V.row( F(i,2) );
          e3.row(i) = V.row( F(i,1) ) - V.row( F(i,0) );

        }

        Eigen::Matrix<DerivedV, Eigen::Dynamic, 3> eX(nf, 3);
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 3> eY(nf, 3);
        eX << e1.col(0), e2.col(0), e3.col(0);
        eY << e1.col(1), e2.col(1), e3.col(1);

        // Extract signed facet areas
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> a = ( e1.col(0).cwiseProduct( e2.col(1) )
            - e2.col(0).cwiseProduct( e1.col(1) ) ) / 2.0;

        // Construct sparse matrix triplets
        typedef Eigen::Triplet<DerivedV> T;
        typedef Eigen::Triplet<std::complex<DerivedV> > TC;

        std::vector<T> tListX, tListY;
        std::vector<TC> tListZ, tListC;

        tListX.reserve( 3 * nf );
        tListY.reserve( 3 * nf );
        tListZ.reserve( 3 * nf );
        tListC.reserve( 3 * nf );

        for( int i = 0; i < nf; i++ ) {
          for( int j = 0; j < 3; j++ ) {

            DerivedV mx = eY(i,j) / ( 2.0 * a(i) );
            DerivedV my = -eX(i,j) / ( 2.0 * a(i) );
            std::complex<DerivedV> mz( mx / 2.0, -my / 2.0 );
            std::complex<DerivedV> mc( mx / 2.0, my / 2.0 );

            tListX.push_back( T( u(i), F(i,j), mx ) );
            tListY.push_back( T( u(i), F(i,j), my ) );
            tListZ.push_back( TC( u(i), F(i,j), mz ) );
            tListC.push_back( TC( u(i), F(i,j), mc ) );

          }
        }

        // Complete sparse operator construction
        Eigen::SparseMatrix<DerivedV> Dx( nf, nv );
        Eigen::SparseMatrix<DerivedV> Dy( nf, nv );
        Eigen::SparseMatrix<std::complex<DerivedV> > Dz( nf, nv );
        Eigen::SparseMatrix<std::complex<DerivedV> > Dc( nf, nv );

        Dx.setFromTriplets( tListX.begin(), tListX.end() );
        Dy.setFromTriplets( tListY.begin(), tListY.end() );
        Dz.setFromTriplets( tListZ.begin(), tListZ.end() );
        Dc.setFromTriplets( tListC.begin(), tListC.end() );

        // Set member variables
        m_Dx = Dx;
        m_Dy = Dy;
        m_Dz = Dz;
        m_Dc = Dc;

      };

  };

} // namespace PlaQuaGE

#endif // _DIFFERENTIAL_OPERATOR_H_
