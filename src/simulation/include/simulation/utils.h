#include <iostream>
#include <cmath>
#include "msgs/action.h"
#include "geometry_msgs/Pose.h"

int getIndex(std::vector<std::string> v, std::string value){
    for(int i = 0; i < v.size(); i++)
    {
        if(v[i].compare(value) == 0)
            return i;
    }
    return -1;
}

float calculate_distance(geometry_msgs::Pose agent_pose, geometry_msgs::Pose enemy_pose){
    float distance_x = agent_pose.position.x - enemy_pose.position.x;
    float distance_y = agent_pose.position.y - enemy_pose.position.y;
    float distance_z = agent_pose.position.z - enemy_pose.position.z;
    float total_distance = pow(distance_x,2) + pow(distance_y,2) + pow(distance_z,2);
    return pow(total_distance,0.5);
}