// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <cmath>
#include <limits>
#include <optional>

#include "eigen_typedefs.h"
#include "quaternion.h"
#include "types.h"
#include "utils.h"

class PnPProblem {
   public:
    struct Parameters {
        CameraState cam;
        RowMajorMatrix3f R;  // Caching the rotation matrix so that we don't
                             // have to recalculate it.
    };

    static constexpr bool kShouldNormalize = false;
    static constexpr int kNumParams = 9;
    static constexpr int kResidualLength = 2;

    PnPProblem(const RefConstRowMajorMatrixX2f &points2D,
               const RefConstRowMajorMatrixX3f &points3D,
               const RefConstArrayXf &weights,
               bool optimize_focal_length = false,
               bool optimize_principal_point = false,
               CameraIntrinsics::Bounds bounds = {})
        : x(points2D),
          X(points3D),
          weights(weights),
          distance_constraints(ArrayXf()),  // Empty by default
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3),
          bounds(bounds) {
        CHECK_EQ(x.rows(), X.rows());
        CHECK(weights.rows() == 0 || weights.rows() == x.rows());
    }
    
    PnPProblem(const RefConstRowMajorMatrixX2f &points2D,
               const RefConstRowMajorMatrixX3f &points3D,
               const RefConstArrayXf &weights,
               const RefConstArrayXf &distance_constraints,
               bool optimize_focal_length = false,
               bool optimize_principal_point = false,
               CameraIntrinsics::Bounds bounds = {})
        : x(points2D),
          X(points3D),
          weights(weights),
          distance_constraints(distance_constraints),
          optimize_focal_length(optimize_focal_length && x.rows() > 3),
          optimize_principal_point(optimize_principal_point && x.rows() > 3),
          bounds(bounds) {
        CHECK_EQ(x.rows(), X.rows());
        CHECK(weights.rows() == 0 || weights.rows() == x.rows());
        CHECK(distance_constraints.rows() == 0 || distance_constraints.rows() == x.rows());
    }

    constexpr size_t NumParams() const { return 9; }
    size_t NumResiduals() const {
        size_t num_reprojection = x.rows();
        // Count valid distance constraints (non-NaN, > 0)
        size_t num_distance = 0;
        if (distance_constraints.rows() > 0) {
            for (size_t i = 0; i < distance_constraints.rows(); ++i) {
                const Float dist = distance_constraints(i);
                if (!std::isnan(dist) && dist > 0.0f) {
                    num_distance++;
                }
            }
        }
        return num_reprojection + num_distance;
    }

    Float Weight(size_t idx) {
        const size_t num_reprojection = x.rows();
        
        // Reprojection residuals use point weights
        if (idx < num_reprojection) {
            if (weights.rows() == 0) {
                return 1.0f;
            } else {
                return weights[idx];
            }
        }
        
        // Distance constraint residuals - use weight 1.0 for now
        // Could be made configurable in the future
        return 1.0f;
    }

    std::optional<Eigen::Vector2f> Evaluate(const Parameters &params,
                                            size_t idx) const {
        const size_t num_reprojection = x.rows();
        
        // Reprojection residuals
        if (idx < num_reprojection) {
            const Eigen::Vector3f Z = params.cam.pose.Apply(X.row(idx));
            if (params.cam.intrinsics.IsBehind(Z)) {
                return Eigen::Vector2f(std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());
            }
            const Eigen::Vector2f z = params.cam.intrinsics.Project(Z);
            return z - Eigen::Vector2f(x.row(idx));
        }
        
        // Distance constraint residuals
        // Find which pin this distance constraint corresponds to
        size_t distance_idx = 0;
        Eigen::Index pin_idx = 0;
        for (Eigen::Index i = 0; i < distance_constraints.rows(); ++i) {
            const Float dist = distance_constraints(i);
            if (!std::isnan(dist) && dist > 0.0f) {
                if (distance_idx == idx - num_reprojection) {
                    pin_idx = i;
                    break;
                }
                distance_idx++;
            }
        }
        
        // Calculate distance from camera to pin in camera space
        // In camera space, camera is at origin, so distance = ||R*X + t||
        const Eigen::Vector3f pin_camera_space = params.cam.pose.Apply(X.row(pin_idx));
        const Float actual_distance = pin_camera_space.norm();
        const Float target_distance = distance_constraints(pin_idx);
        const Float distance_error = actual_distance - target_distance;
        
        // Return as 2D vector: [distance_error, 0] for compatibility
        return Eigen::Vector2f(distance_error, 0.0f);
    }

    bool EvaluateWithJacobian(const Parameters &params, size_t idx,
                              RowMajorMatrixf<kResidualLength, kNumParams> &J,
                              Eigen::Vector2f &res) const {
        const size_t num_reprojection = x.rows();
        
        // Reprojection residuals
        if (idx < num_reprojection) {
            const Pose &pose = params.cam.pose;
            const CameraIntrinsics &intrin = params.cam.intrinsics;
            const RowMajorMatrix3f &R = params.R;

            const Eigen::Vector3f Z = X.row(idx);

            Eigen::Vector3f RtZ;
            RowMajorMatrixf<3, 3> dRtZ_dR;
            // dRtZ_dt is Identity, so we can drop it
            // RowMajorMatrixf<3, 3> dRtZ_dt;
            Pose::ApplyWithJac(Z, R, pose.t, &RtZ, nullptr, &dRtZ_dR, nullptr);

            Eigen::Vector2f z;
            RowMajorMatrixf<2, 3> dz_dRtZ;
            RowMajorMatrixf<2, 3> dz_dIntrin;
            intrin.ProjectWithJac(RtZ, &z, &dz_dRtZ, &dz_dIntrin);

            // Residual
            res = z - Eigen::Vector2f(x.row(idx));

            // Jacobian
            J.block<2, 3>(0, 0) = dz_dRtZ * dRtZ_dR;
            J.block<2, 3>(0, 3) = dz_dRtZ;

            J.block<2, 3>(0, 6) = dz_dIntrin;
            if (!optimize_focal_length) {
                J.block<2, 1>(0, 6).setZero();
            }
            if (!optimize_principal_point) {
                J.block<2, 2>(0, 7).setZero();
            }

            return true;
        }
        
        // Distance constraint residuals
        // Find which pin this distance constraint corresponds to
        size_t distance_idx = 0;
        Eigen::Index pin_idx = 0;
        for (Eigen::Index i = 0; i < distance_constraints.rows(); ++i) {
            const Float dist = distance_constraints(i);
            if (!std::isnan(dist) && dist > 0.0f) {
                if (distance_idx == idx - num_reprojection) {
                    pin_idx = i;
                    break;
                }
                distance_idx++;
            }
        }
        
        // Calculate distance from camera to pin in camera space
        // In camera space, camera is at origin, so distance = ||R*X + t||
        const Eigen::Vector3f pin_camera_space = params.cam.pose.Apply(X.row(pin_idx));
        const Float actual_distance = pin_camera_space.norm();
        const Float target_distance = distance_constraints(pin_idx);
        const Float distance_error = actual_distance - target_distance;
        
        // Residual: [distance_error, 0]
        res = Eigen::Vector2f(distance_error, 0.0f);
        
        // Jacobian for distance constraint: d(||pin_camera_space|| - target) / d(params)
        // pin_camera_space = R * X + t, so ||pin_camera_space|| = ||R * X + t||
        // d(||v||)/dv = v / ||v|| (normalized vector)
        if (actual_distance > 1e-6f) {
            const Eigen::Vector3f normalized = pin_camera_space / actual_distance;
            
            // d(distance)/dR: normalized^T * d(R*X)/dR
            // d(R*X)/dR = [d(R*X)/dq] where q is quaternion
            const Eigen::Vector3f X_local = X.row(pin_idx);
            RowMajorMatrixf<3, 3> dRtZ_dR;
            Eigen::Vector3f RtZ;
            Pose::ApplyWithJac(X_local, params.R, params.cam.pose.t, &RtZ, nullptr, &dRtZ_dR, nullptr);
            
            // Jacobian: [normalized^T * dRtZ_dR, normalized^T, 0, 0, 0]
            J.setZero();
            J.block<1, 3>(0, 0) = normalized.transpose() * dRtZ_dR;  // d/dq (rotation)
            J.block<1, 3>(0, 3) = normalized.transpose();            // d/dt (translation)
            // Distance doesn't depend on intrinsics
            J.block<1, 3>(0, 6).setZero();
            // Second row is zero (padding)
            J.block<1, 9>(1, 0).setZero();
        } else {
            // Degenerate case: pin is at camera origin
            J.setZero();
        }
        
        return true;
    }

    void Step(const Parameters &params,
              const RowMajorMatrixf<kNumParams, 1> &dp,
              Parameters &result) const {
        const CameraState &camera = params.cam;
        CameraState &camera_new = result.cam;

        camera_new.pose.q = QuatStepPost(camera.pose.q, dp.block<3, 1>(0, 0));
        camera_new.pose.t = camera.pose.t + dp.block<3, 1>(3, 0);

        if (optimize_focal_length) {
            camera_new.intrinsics.fy = camera.intrinsics.fy + dp(6, 0);
            camera_new.intrinsics.fx =
                camera_new.intrinsics.fy * camera_new.intrinsics.aspect_ratio;

            camera_new.intrinsics.fy = std::clamp(camera_new.intrinsics.fy,
                                                  bounds.f_low, bounds.f_high);
            camera_new.intrinsics.fx = std::clamp(camera_new.intrinsics.fx,
                                                  bounds.f_low, bounds.f_high);
        }
        if (optimize_principal_point) {
            camera_new.intrinsics.cx = camera.intrinsics.cx + dp(7, 0);
            camera_new.intrinsics.cy = camera.intrinsics.cy + dp(8, 0);

            camera_new.intrinsics.cx = std::clamp(
                camera_new.intrinsics.cx, bounds.cx_low, bounds.cx_high);
            camera_new.intrinsics.cy = std::clamp(
                camera_new.intrinsics.cy, bounds.cy_low, bounds.cy_high);
        }

        result.R = result.cam.pose.R();
    }

   private:
    const RefConstRowMajorMatrixX2f x;
    const RefConstRowMajorMatrixX3f X;
    const RefConstArrayXf weights;
    const ArrayXf distance_constraints;  // Store as ArrayXf, not Ref, to allow empty default

    const bool optimize_focal_length;
    const bool optimize_principal_point;

    const CameraIntrinsics::Bounds bounds;
};
