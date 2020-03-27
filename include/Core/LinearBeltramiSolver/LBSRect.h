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

#ifndef _LBS_RECT_H_
#define _LBS_RECT_H_

#include <complex>
#include <vector>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <igl/boundary_loop.h>
#include <igl/LinSpaced.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/ismember.h>
#include <igl/slice_into.h>
#include <igl/speye.h>
#include <igl/find.h>

#include "LinearBeltramiSolver.h"
#include "../General/Generalized_Laplacian.h"
#include "../General/circshift.h"

namespace PlaQuaGE {

  ///
  /// A Linear Beltrami Solver for problems on the canonical domain of a rectangle.  In other
  /// words, in addition to whatever landmark/target correspondences that may exist in the
  /// bulk of the mesh, the real parts of the left and right sides of the image region must be
  /// specified and the imaginary parts of the top and bottom sides of the image region must be
  /// specified.  Note that this construction fixes the corners of the image region.  Detailed
  /// description of the algorithm can be found in Lam and Lui (2014).
  ///
  /// Templates:
  ///   DerivedV    Derived data type of Eigen matrix for V (e.g. double from MatrixXd)
  ///   DerivedF    Derived data type of Eigen matrix for F (e.g. int from MatrixXi)
  ///
  template <typename DerivedV, typename DerivedF>
  class LBSRect : public LinearBeltramiSolver<DerivedV, DerivedF> {

    private:

      // All target vertex indices of the X-subproblem (includes boundary constraints)
      Eigen::VectorXi m_landmarkIDxX;

      // All target vertex indices of the Y-subproblem (includes boundary constraints)
      Eigen::VectorXi m_landmarkIDxY;

      // All target vertex positions of the X-subproblem (includes boundary constraints)
      Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> m_targetX;

      // All target vertex positions of the Y-subproblem (includes boundary constraints)
      Eigen::Matrix<DerivedV, Eigen::Dynamic, 1> m_targetY;

      // The X-bounds of the image region
      Eigen::Matrix<DerivedV, 2, 1> m_xlim;

      // The Y-bounds of the image region
      Eigen::Matrix<DerivedV, 2, 1> m_ylim;

      // The vertex indices of the corners of the image region
      // NOTE: CORNERS SHOULD BE ORDERED [ BL BR TR TL ]
      Eigen::VectorXi m_corners;

    public:

      ///
      /// Basic Constructor
      ///
      /// Inputs:
      ///   V         #V by 2 list of mesh vertex positions
      ///   F         #F by 3 list of mesh faces
      ///   landmark  #C by 2 list of the real coordinates of the landmark points
      ///   target    #C by 2 list of the real coordinates of the target points
      ///   xlim      2 by 1 list of the x-bounds of the image region
      ///   ylim      2 by 1 list of the y bounds of the image region
      ///   corners   4 by 1 list of the vertex indices of the corners of the image region
      ///             NOTE: CORNERS SHOULD BE ORDERED [ BL BR TR TL ]
      ///
      LBSRect( const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &V,
          const Eigen::Matrix<DerivedF, Eigen::Dynamic, 3> &F,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &landmark,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &target,
          const Eigen::Matrix<DerivedV, 2, 1> &xlim,
          const Eigen::Matrix<DerivedV, 2, 1> &ylim,
          const Eigen::VectorXi corners ) :
        LinearBeltramiSolver( V, F ), m_xlim( xlim ), m_ylim( ylim ),
        m_corners( corners ) {

          Process_Landmarks( landmark, target );

        };

    private:

      ///
      /// Divide the boundary loop into segments based on the supplied corner indices
      ///
      /// Inputs:
      ///   bdy     #BV by 1 ordered loop of boundary vertex indices
      ///   corners #C by 1 ordered list of the vertex indices of the
      ///           corners of the image region
      ///
      ///
      /// Outputs:
      ///   ccBdy   A vector containing the boundary loop division segments
      ///
      void Closed_Curve_Division( const Eigen::VectorXi &bdy,
          const Eigen::VectorXi &corners, std::vector<Eigen::VectorXi> &ccBdy ) {

        // Find the index of the first corner point in the boundary loop
        int firstIndex;
        for( int i = 0; i < bdy.size(); i++ ) {
          if ( bdy(i) == corners(0) ) {
            firstIndex = bdy(i);
            break;
          }
        }

        // Shift boundary so the first corner is first in the list
        Eigen::VectorXi bdyShift( bdy.size() );
        circshift( bdyShift, bdy, -firstIndex );

        // Ensure the ordering of the boundary list is consistent
        Eigen::VectorXi cornerIDx( corners.size() );
        igl::find( corners, cornerIDx, _ );

        bool goodOrder = true;
        for( int i = 1; i < corners.size(); i++ ) {
          if ( cornerIDx( i ) <= cornerIDx( i-1 ) ) {
            goodOrder = false;
            break;
          }
        }

        Eigen::VectorXi bdyFinal( bdy.size() );
        if ( !goodOrder ) {
          bdyFinal = bdyShift.colwise().reverse().eval();
        } else {
          bdyFinal = bdyShift;
        }

        // Add the divisions to the output vector
        ccBdy.clear();
        ccBdy.reserve( corners.size() );

        for( i = 1; i < corners.size(); i++ ) {
          ccBdy.push_back( bdyFinal.segment( cornerIDx(i-1),
                ( cornerIDx(i)-cornerIDx(i-1)+1 ) ) );
        }

        Eigen::VectorXi finalSegment( corners.size()-cornerIDx( corners.size() ) + 2 );
        finalSegment << bdyFinal.tail( corners.size()-cornerIDx( corners.size() ) + 1 ),
                     bdyFinal(0);
        ccBdy.push_back( finalSegment );

      };


      ///
      /// Process the user supplied landmark/target correspondences.  All boundary vertices
      /// will be forced to conform to the rectangular boundary condition
      ///
      /// Inputs:
      ///   landmark    #C by 2 list of the real coordinates of the landmark points
      ///   target      #C by 2 list of the real coordinates of the target points
      ///
      void Process_Landmarks(
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &landmark,
          const Eigen::Matrix<DerivedV, Eigen::Dynamic, 2> &target ) {

        // Find the closest vertices to the input landmark points
        Eigen::VectorXi inputIDx, vIDx;
        vIDx = igl::LinSpaced<int>( m_V.rows(), 0, (m_V.rows()-1) );
        igl::point_mesh_squared_distance( landmark, m_V, vIDx, _, inputIDx, _ );

        // Get the boundary vertex IDs
        // NOTE: ASSUMES THE MESH IS A TOPOLOGICAL DISK
        Eigen::VectorXi bdyIDx;
        igl::boundary_loop( m_F, bdyIDx );

        // Determine if any boundary vertices overlap with the user input
        Eigen::Matrix<bool, Eigen::Dynamic, 1> ia;
        igl::ismember( inputIDx, bdyIDx, ia, _ );

        // Count the overlap of the boundary with the user input
        int count = 0;
        for( int i = 0; i < bdyIDx.size(); i++ )
          if ( ia(i) )
            count++;

        // Get the closed curve division of the boundary
        // ccBdy[0] = bottom side
        // ccBdy[1] = right side
        // ccBdy[2] = top side
        // ccBdy[3] = left side
        std::vector<Eigen::VectorXi> ccBdy;
        Closed_Curve_Division( boundary_loop, m_corners, ccBdy );

        // Set the basic landmark/target structures ignoring boundary vertices
        int n = inputIDx.size() - count;
        m_landmarkIDx.resize( n, Eigen::NoChange );
        m_targetxy.resize( n, Eigen::NoChange );
        m_targetc.resize( n, Eigen::NoChange );

        int count2 = 0;
        for( int k = 0; k < inputIDx.size(); k++ ){
          if ( !ia(k) ) {

            m_landmarkIDx(count2) = inputIDx(k);
            m_targetxy.row(count2) = target.row(k);
            m_targetc(count2) = target(k,1) + 1i * target(k,2);

          }
        }

        // Specify the X-subproblem
        int nX = n + ccBdy[1].size() + ccBdy[3].size();
        m_landmarkIDxX.resize( nX, Eigen::NoChange );
        m_targetX.resize( nX, Eigen::NoChange );

        Eigen::Matrix<DerivedV, nX, 1> leftSide;
        leftSide = Eigen::Matrix<DerivedV, nX, 1>::Constant( m_xlim(0) );

        Eigen::Matrix<DerivedV, nX, 1> rightSide;
        rightSide = Eigen::Matrix<DerivedV, nX, 1>::Constant( m_xlim(1) );

        m_landmarkIDxX << m_landmarkIDx, ccBdy[1], ccBdy[3];
        m_targetX << m_targetxy.col(0), rightSide, leftSide;

        // Specify the Y-subproblem
        int nY = n + ccBdy[0].size() + ccBdy[2].size();
        m_landmarkIDxY.resize( nY, Eigen::NoChange );
        m_targetY.resize( nY, Eigen::NoChange );

        Eigen::Matrix<DerivedV, nY, 1> bottomSide;
        bottomSide = Eigen::Matrix<DerivedV, nY, 1>::Constant( m_ylim(0) );

        Eigen::Matrix<DerivedV, nY, 1> topSide;
        topSide = Eigen::Matrix<DerivedV, nY, 1>::Constant( m_ylim(1) );

        m_landmarkIDxY << m_landmarkIDx, ccBdy[0], ccBdy[2];
        m_targetY << m_targetxy.col(1), bottomSide, topSide;

      };
