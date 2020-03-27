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

#ifndef _LBS_DIRICHLET_H_
#define _LBS_DIRICHLET_H_

#include <complex>
#include <stdexcept>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <igl/boundary_loop.h>
#include <igl/LinSpaced.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/ismember.h>
#include <igl/slice_into.h>
#include <igl/speye.h>
#include <igl/slice.h>

#include "LinearBeltramiSolver.h"
#include "../General/Generalized_Laplacian.h"

namespace PlaQuaGE {

  ///
  /// A Linear Beltrami Solver for problems with fully specified Dirichlet boundary
  /// conditions.  In other words, in addition to whatever landmark/target correspondences
  /// that may exist in the bulk of the mesh, the target positions of all boundary
  /// vertices under mapping must be specified.  In principle this should make the solver
  /// valid for an arbitrary Jordan region. Detailed description of the algorithm can be
  /// found in Lam and Lui (2014).
  ///
  /// Templates:
  ///   DerivedV    Derived data type of Eigen matrix for V (e.g. double from MatrixXd)
  ///   DerivedF    Derived data type of Eigen matrix for F (e.g. int from MatrixXi)
  ///
  template <typename DerivedV, typename DerivedF>
  class LBSDirichlet : public LinearBeltramiSolver<DerivedV, DerivedF> {

    public:

      ///
      /// Basic constructor
      ///
      /// Inputs:
      ///   V           #V by 2 list of mesh vertex positions
      ///   F           #F by 3 list of mesh faces
      ///   landmark    #C by 2 list of the real coordinates of the landmark points
      ///   target      #C by 2 list of the real coordinates of the target points
      ///
      LBSDirichlet( const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &landmark,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &target ) :
        LinearBeltramiSolver<DerivedV, DerivedF>( V, F ) {

          Process_Landmarks( landmark, target );

      };

    private:

      ///
      /// Process the user supplied landmark/target correspondences.  All boundary
      /// vertices whose target positions are not explicitly specified by the user
      /// are held fixed.
      ///
      /// Inputs:
      ///   landmark    #C by 2 list of the real coordinates of the landmark points
      ///   target      #C by 2 list of the real coordinates of the target points
      ///
      void Process_Landmarks(
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &landmark,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &target ) {

        // Find the closest vertices to the input landmark points
        Eigen::VectorXi inputIDx( landmark.rows() );
        // THIS IS PROBABLY REALLY SLOW - TRY TO FIX
        for( int i = 0; i < landmark.rows(); i++ ) {

          Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> curLM( this->m_V.rows(), 2 );
          curLM = landmark.row(i).colwise().replicate( this->m_V.rows() );

          Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> curDist( curLM.rows() );
          curDist = ( this->m_V - curLM ).rowwise().squaredNorm();

          int minID = 0;
          DerivedV minVal = curDist(0);
          for( int k = 1; k < this->m_V.rows(); k++ ) {
            if ( curDist(k) < minVal ) {
              minID = k;
              minVal = curDist(k);
            }
          }

          inputIDx(i) = minID;

        }

        // Get the boundary vertex IDs
        // NOTE: ASSUMES MESH IS A TOPOLOGICAL DISK
        Eigen::VectorXi bdyIDx;
        igl::boundary_loop( this->m_F, bdyIDx );

        // Determine which boundary vertices should be set by user input
        Eigen::Matrix<bool, Eigen::Dynamic, 1> ia;
        Eigen::MatrixXi locb; // NOT USED - TRY TO FIX
        igl::ismember( bdyIDx, inputIDx, ia, locb );

        // Get the boundary vertex positions and
        // count the overlap of the boundary with the user input
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> bdyLoc( bdyIDx.size(), 2 );
        int count = 0;
        for( int i = 0; i < bdyIDx.size(); i++ ) {

          bdyLoc.row(i) = this->m_V.row( bdyIDx(i) );

          if ( ia(i) ) count++;

        }

        // Assemble complete landmark/target structures
        int n = inputIDx.size() + bdyIDx.size() - count;
        this->m_landmarkIDx.resize( n, Eigen::NoChange );
        this->m_targetxy.resize( n, Eigen::NoChange );
        this->m_targetc.resize( n, Eigen::NoChange );

        for( int k = 0; k < inputIDx.size(); k++ ) {

          this->m_landmarkIDx(k) = inputIDx(k);
          this->m_targetxy.row(k) = target.row(k);
          this->m_targetc(k) = std::complex<DerivedV>( target(k,0), target(k,1) );

        }

        int count2 = inputIDx.size();
        for( int k = 0; k < bdyIDx.size(); k++ ) {
          if ( !ia(k) ) {

            this->m_landmarkIDx(count2) = bdyIDx(k);
            this->m_targetxy.row(count2) = bdyLoc.row(k);
            this->m_targetc(count2) = std::complex<DerivedV>( bdyLoc(k,0), bdyLoc(k,1) );
            count2++;

          }
        }

      };

    public:

      ///
      /// Solve the Beltrami equation given complete Dirichlet boundary conditions
      ///
      /// Inputs:
      ///   tarMu   #F by 1 list of the target Beltrami coefficient defined on faces
      ///
      /// Outputs:
      ///   map     #V by 2 list of the mapped vertex coordinates
      ///   mapMu   #F by 1 list of the actual Beltrami coefficient derived from the map
      ///
      void Solve(
          const Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &tarMu,
          Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &map,
          Eigen::Matrix<std::complex<DerivedV>, Eigen::Dynamic, 1> &mapMu ) {

        // Construct the generalized Laplacian
        Eigen::SparseMatrix<DerivedV> A =
          Generalized_Laplacian<DerivedV, DerivedF>::Build( this->m_V, this->m_F, tarMu );

        // Construct the constraint vector
        Eigen::VectorXi vIDx( this->m_V.rows() );
        vIDx = igl::LinSpaced<Eigen::VectorXi>( this->m_V.rows(), 0, (this->m_V.rows()-1) );
        Eigen::SparseMatrix<DerivedV> Asub;

        igl::slice( A, vIDx, this->m_landmarkIDx, Asub );

        Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> bx = -Asub * this->m_targetxy.col(0);
        Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> by = -Asub * this->m_targetxy.col(1);

        for( int i = 0; i < this->m_landmarkIDx.size(); i++ ) {
          bx( this->m_landmarkIDx( i ) ) = this->m_targetxy(i,0);
          by( this->m_landmarkIDx( i ) ) = this->m_targetxy(i,1);
        }

        // Set all rows and columns corresponding to landmark vertices equal to zero
        for( int k = 0; k < this->m_landmarkIDx.size(); k++ ) {
          int rmInd = this->m_landmarkIDx(k);
          A.prune( [rmInd](int i, int j, DerivedV) { return i!=rmInd && j!=rmInd; } );
        }

        // Set the landmark diagonals equal to one
        for( int k = 0; k < this->m_landmarkIDx.size(); k++ ) {
          int lmInd = this->m_landmarkIDx(k);
          A.coeffRef( lmInd, lmInd ) = (DerivedV) 1.0;
        }

        A.makeCompressed();


        // Solve the Beltrami equation
        /*
        Eigen::SparseQR<Eigen::SparseMatrix<DerivedV>, Eigen::COLAMDOrdering<int> > solver;
        solver.setPivotThreshold(0.0); // Matrix is full-rank
        solver.compute( A );
        */

        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<DerivedV> > solver( A );
        if ( solver.info() != Eigen::Success ) {
          // Decomposition failed
          std::runtime_error( "Eigen decomposition failed" );
        }

        Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> mapX, mapY;

        mapX = solver.solve( bx );
        if( solver.info() != Eigen::Success ) {
          // Solution failed
          std::runtime_error( "Eigen LLSQ solution failed" );
        }

        mapY = solver.solve( by );
        if( solver.info() != Eigen::Success ) {
          // Solution failed
          std::runtime_error( "Eigen LLSQ solution failed" );
        }

        /*
        // Overwrite the input map
        for( int i = 0; i < map.rows(); i++ ) {
          map(i,0) = mapX(i);
          map(i,1) = mapY(i);
        }
        */

        map << mapX, mapY;

        // Overwrite the input map Beltrami coefficient
        this->Beltrami_From_Map( map, mapMu );

      };

  };

} // namespace PlaQuaGE

#endif // _LBS_DIRICHLET_H_
