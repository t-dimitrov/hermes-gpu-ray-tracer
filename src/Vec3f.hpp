/*
Taken from https://raytracing.github.io/books/RayTracingInOneWeekend.html#thevec3class
*/

#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include <cuda_runtime.h>

namespace Hermes
{
    class Vec3f {
    public:
        float e[3];

        __host__ __device__ Vec3f() : e{ 0,0,0 } {}
        __host__ __device__ Vec3f(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

        __host__ __device__ float x() const { return e[0]; }
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }

        __host__ __device__ Vec3f operator-() const { return Vec3f(-e[0], -e[1], -e[2]); }
        __host__ __device__ float operator[](int i) const { return e[i]; }
        __host__ __device__ float& operator[](int i) { return e[i]; }

        __host__ __device__ Vec3f& operator+=(const Vec3f& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ Vec3f& operator*=(float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__ Vec3f& operator/=(float t) {
            return *this *= 1 / t;
        }

        __host__ __device__ float Length() const
        {
            return std::sqrt(LengthSquared());
        }

        __host__ __device__ float LengthSquared() const
        {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        // Returns true if the vector is close to zero in all dimensions
        __host__ __device__ bool IsNearZero() const
        {
            float s = 1e-8f;
            return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
        }
    };

    // Point3 is just an alias for Vec3f, but useful for geometric clarity in the code.
    using Point3f = Vec3f;
    using Color3f = Vec3f;

    // Vector Utility Functions

    inline __host__ __device__ std::ostream& operator<<(std::ostream& out, const Vec3f& v) {
        return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
    }

    inline __host__ __device__ Vec3f operator+(const Vec3f& u, const Vec3f& v) {
        return Vec3f(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
    }

    inline __host__ __device__ Vec3f operator-(const Vec3f& u, const Vec3f& v) {
        return Vec3f(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
    }

    inline __host__ __device__ Vec3f operator*(const Vec3f& u, const Vec3f& v) {
        return Vec3f(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
    }

    inline __host__ __device__ Vec3f operator*(float t, const Vec3f& v) {
        return Vec3f(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    inline __host__ __device__ Vec3f operator*(const Vec3f& v, float t) {
        return t * v;
    }

    inline __host__ __device__ Vec3f operator/(const Vec3f& v, float t) {
        return (1 / t) * v;
    }

    inline __host__ __device__ Vec3f Normalize(const Vec3f& v)
    {
        float mag = v.Length();
        if (mag == 0.0f) // In case of a zero vector
        {
            return Vec3f(0.0f, 0.0f, 0.0f);
        }
        return Vec3f(v.x() / mag, v.y() / mag, v.z() / mag);
    }

    inline __host__ __device__ float Dot(const Vec3f& u, const Vec3f& v)
    {
        return u.e[0] * v.e[0]
            + u.e[1] * v.e[1]
            + u.e[2] * v.e[2];
    }

    inline __host__ __device__ Vec3f Cross(const Vec3f& u, const Vec3f& v)
    {
        return Vec3f(u.e[1] * v.e[2] - u.e[2] * v.e[1],
            u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }

    inline __host__ __device__ Vec3f UnitVector(const Vec3f& v)
    {
        return v / v.Length();
    }

    inline __host__ __device__ Vec3f Reflect(const Vec3f& I, const Vec3f& N)
    {
        return I - 2.0f * Dot(I, N) * N;
    }

    inline __host__ __device__ Vec3f Refract(const Vec3f& uv, const Vec3f& N, float etaiOverEtat)
    {
        float cosTheta = std::fmin(Dot(-uv, N), 1.0f);
        Vec3f rOutPerp = etaiOverEtat * (uv + cosTheta * N);
        Vec3f rOutParallel = -std::sqrt(std::fabs(1.0f - rOutPerp.LengthSquared())) * N;
        return rOutPerp + rOutParallel;
    }
}

#endif