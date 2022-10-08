#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <vector>
#include <random>
#include "msgs/action.h"
#include "msgs/reset.h"

int border_check(float x, float y, float z){
    if(x<0.0 || x>25.0 || y<0.0 || y>25.0 || z<0.0 || z>25.0){
        return 0;
    }
    else{
        return 1;
    }
}


int is_done(geometry_msgs::Pose agent_pose, geometry_msgs::Pose enemy_pose){
    double diff_x = abs(agent_pose.position.x - enemy_pose.position.x);
    double diff_y = abs(agent_pose.position.y - enemy_pose.position.y);
    double diff_z = abs(agent_pose.position.z - enemy_pose.position.z);

    int is_in_border_agent = border_check(agent_pose.position.x, agent_pose.position.y, agent_pose.position.z);
    int is_in_border_enemy = border_check(enemy_pose.position.x, enemy_pose.position.y, enemy_pose.position.z);

    if(diff_x < 0.5 && diff_y < 0.5 && diff_z < 0.5){
        return 1;
    }
    else if(is_in_border_agent == 0 || is_in_border_enemy == 0){
        return 0;
    }
    else{
        return -1;
    }
}


gazebo_msgs::ModelState init_state(std::string model_name){
    gazebo_msgs::ModelState state;
    state.model_name = model_name;

    int random_int_x = rand() % 25;
    double random_x = (double)random_int_x;
    int random_int_y = rand() % 25;
    double random_y = (double)random_int_y;
    int random_int_z = rand() % 25;
    double random_z = (double)random_int_z;
    
    state.pose.position.x = random_x;
    state.pose.position.y = random_y;
    state.pose.position.z = random_z;

    return state;
}