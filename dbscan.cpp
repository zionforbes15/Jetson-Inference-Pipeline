#include "dbscan.h"

int DBSCAN::run() {
    int clusterID = 1;
    for (auto& point : m_points) {
        if (point.clusterID == UNCLASSIFIED) {
            if (expandCluster(point, clusterID) != FAILURE) {
                clusterID += 1;
            }
        }
    }
    return 0;
}

int DBSCAN::expandCluster(PointZX point, int clusterID)
{
	std::vector<int> clusterSeeds = calculateCluster(point);

	if (clusterSeeds.size() < m_minPoints)
	{
		point.clusterID = NOISE;
		return FAILURE;
	}
	else
	{
		int index = 0, indexCorePoint = 0;
		std::vector<int>::iterator iterSeeds;
		for (iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
		{
			m_points.at(*iterSeeds).clusterID = clusterID;
			if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y)
			{
				indexCorePoint = index;
			}
			++index;
		}
		clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);

		for (std::vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i)
		{
			std::vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

			if (clusterNeighors.size() >= m_minPoints)
			{
				std::vector<int>::iterator iterNeighors;
				for (iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors)
				{
					if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE)
					{
						if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED)
						{
							clusterSeeds.push_back(*iterNeighors);
							n = clusterSeeds.size();
						}
						m_points.at(*iterNeighors).clusterID = clusterID;
					}
				}
			}
		}

		return SUCCESS;
	}
}

std::vector<int> DBSCAN::calculateCluster(PointZX point)
{
	int index = 0;
	std::vector<PointZX>::iterator iter;
	std::vector<int> clusterIndex;
	for (iter = m_points.begin(); iter != m_points.end(); ++iter)
	{
		//std::cout << "calculateDistance(point, *iter) " << calculateDistance(point, *iter) <<" "<< m_epsilon << std::endl;
		if (calculateDistance(point, *iter) <= m_epsilon)
		{
			clusterIndex.push_back(index);
		}
		index++;
	}
	return clusterIndex;
}
