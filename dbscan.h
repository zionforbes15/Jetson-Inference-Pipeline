#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include <iostream>

typedef struct Point_zx
{
    float x, y, z;  // 坐标
    int clusterID;  // 聚类后的 ID
} PointZX;

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

class DBSCAN {
public:
    DBSCAN(int minPts, float eps, std::vector<PointZX> points) {
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
    }
    ~DBSCAN() {}

    int run();
    std::vector<int> calculateCluster(PointZX point);
    int expandCluster(PointZX point, int clusterID);
    inline double calculateDistance(const PointZX& p1, const PointZX& p2) {
        return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2);
    }

public:
    std::vector<PointZX> m_points;

private:
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

#endif