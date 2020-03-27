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

#ifndef _LINEAR_BELTRAMI_SOLVER_H_
#define _LINEAR_BELTRAMI_SOLVER_H_

#include <complex>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "../General/DifferentialOperator.h"

namespace PlaQuaGE {

  ///
  /// An abstract base class for the Linear Beltrami Solver.  For a given piecewise
  /// constant Beltrami coefficent defined on the faces of an input triangulation,
  /// the children of this class are able to solve the corresponding Beltrami equation
  /// via a linear system to determine the mapped position of the input vertices.
  ///
  /// Templates:
  ///   DerivedV    Derived data type of Eigen matrix for V (e.g. double from MatrixXd)
  ///   DerivedF    Derived data type of Eigen matrix for F (e.g. int from MatrixXi)
  ///
  template <typename DerivedV, typename DerivedF>
  class LinearBeltramiSolver {

    protected:

      const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> m_V; // Vertex coordinate list
      const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> m_F; // Face connectivity list

      // Differential operator defined on input mesh
      DifferentialOperator<DerivedV, DerivedF> m_diffOp;

      // The vertex indices of those vertices in the original
      // mesh closest to the input landmark points
      Eigen::VectorXi m_landmarkIDx;

      // The real coordinates of the target positions
      Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> m_targetxy;

      // The complex representation of the target positions
      Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1 > m_targetc;

    public:

      ///
      /// Basic constructor
      ///
      /// Inputs:
      ///   V   #V by 2 list of mesh vertex positions
      ///   F   #F by 3 list of mesh faces
      ///
      LinearBeltramiSolver( const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F ) : m_V( V ), m_F( F ) {

        m_diffOp = DifferentialOperator<DerivedV, DerivedF>( V, F );

      };

      ///
      /// Solve the Beltrami equation.
      ///
      /// Inputs:
      ///   tarMu   #F by 1 list of the target Beltrami coefficient defined on faces
      ///
      /// Outputs:
      ///   map     #V by 2 list of the mapped vertex coordinates
      ///   mapMu   #F by 1 list of the actual Beltrami coefficient derived from the map
      ///
      virtual void Solve(
          const Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &tarMu,
          Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &map,
          Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mapMu ) = 0;


      ///
      /// Overloaded function to calculate the Beltrami coefficient for a given map
      ///
      /// Inputs:
      ///   mapC  #V by 1 list of the complex mapped vertex coordinates
      ///
      /// Outputs:
      ///   mu    #F by 1 list of the Beltrami coefficients defined on faces
      ///
      void Beltrami_From_Map(
          const Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mapC,
          Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mu ) {

        Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> DfDz, DfDc;

        DfDz = m_diffOp.Dz() * mapC;
        DfDc = m_diffOp.Dc() * mapC;

        mu = DfDc.cwiseQuotient( DfDz );

      };

      ///
      /// Overloaded function to calculate the Beltrami coefficient for a given map
      ///
      /// Inputs:
      ///   map   #V by 2 list of the real mapped vertex coordinates
      ///
      /// Outputs:
      ///   mu    #F by 1 list of the Beltrami coefficients defined on faces
      ///
      void Beltrami_From_Map(
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &map,
          Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mu ) {

        Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> mapC(map.rows(), 1);
        for( int v = 0; v < map.rows(); v++ ) {
          mapC(v) = std::complex<DerivedV>( map(v,0), map(v,1) );
        }

        Beltrami_From_Map( mapC, mu );

      };

      ///
      /// Truncates the magnitude of the Beltrami coefficents
      /// greater than a given bound down to a given constant
      ///
      /// Inputs:
      ///   mu         #F by 1 list of the Beltrami coefficients defined on faces
      ///   bound      The bound at which the Beltrami coefficient is chopped
      ///   constant   The constant to which the Beltrami coefficient is chopped down
      ///
      /// Outputs:
      ///   mu        The chopped Beltrami coefficient list
      ///
      void Beltrami_Chop(
          Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mu,
          DerivedV bound = 1.0,
          DerivedV constant = 1.0 ) {

        for( int f = 0; f < mu.size(); f++ )
          if ( std::abs( mu(f) ) >= bound )
            mu(f) = constant * std::exp( std::complex<DerivedV>( 0.0, std::arg( mu(f) ) ) );

      };

      ///
      /// Determines the number of Beltrami coefficients with a magnitude greater than
      /// or equal to a given bound
      ///
      /// Inputs:
      ///   mu      #F by 1 list of the Beltrami coefficients defined on faces
      ///   bound   The bound at which the function is triggered
      ///
      /// Outputs:
      ///   count:  The number of Beltrami coefficients greater than the given bound
      ///
      int Beltrami_Overlap(
          const Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mu,
          DerivedV bound = 1.0 ) {

        int count = 0;
        for( int f = 0; f < mu.size(); f++ )
          if( std::abs( mu(f) ) >= bound )
            count++;

        return count;

      };

      ///
      /// Get the landmark vertex IDs
      ///
      Eigen::VectorXi Landmarks() const { return m_landmarkIDx; };

      ///
      /// Get the real coordinates of the target points
      ///
      Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> Targets() const { return m_targetxy; };

    private:

      ///
      /// Process the user supplied landmark/target correspondences
      ///
      /// Inputs:
      ///   landmark    #C by 2 list of the real coordinates of the landmark points
      ///   target      #C by 2 list of the real coordinates of the target points
      ///
      virtual void Process_Landmarks(
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &landmark,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &target ) = 0;

  };

} // namespace PlaQuaGE

#endif // _LINEAR_BELTRAMI_SOLVER_H_
