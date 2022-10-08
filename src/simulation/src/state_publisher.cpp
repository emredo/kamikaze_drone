#include <iostream>
#include <thread>

#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "gazebo_msgs/ModelStates.h"
#include "gazebo_msgs/SetModelState.h"
#include "std_srvs/Empty.h"
#include "msgs/state.h"
#include "msgs/action.h"
#include "msgs/observation.h"
#include "simulation/borders.h"
#include "simulation/utils.h"
#include "simulation/stateOps.h"

std_srvs::Empty resetSimSrv;
gazebo_msgs::ModelState target_action;
geometry_msgs::Pose agent_pose;
geometry_msgs::Pose enemy_pose;
geometry_msgs::Twist agent_twist;
geometry_msgs::Twist enemy_twist;
msgs::reset reset;

float MAX_VEL = 5;
float reward;
int done_indice;
float SUCCESS_REWARD = 300.0;
float REWARD = 5.0;
int done_status;
bool is_action = false;
float distance;
bool stopper = false;

void stateCallback(gazebo_msgs::ModelStates data){
    int agent_index = getIndex(data.name, "agent");
    int enemy_index = getIndex(data.name, "enemy");
    agent_pose = data.pose[agent_index];
    agent_twist = data.twist[agent_index];
    enemy_pose = data.pose[enemy_index];
    enemy_twist = data.twist[enemy_index];
    target_action.pose = agent_pose;
}

void actionCallback(msgs::action data){
    target_action.twist.linear.x = data.ball1_vx * MAX_VEL;
    target_action.twist.linear.y = data.ball1_vy * MAX_VEL;
    target_action.twist.linear.z = data.ball1_vz * MAX_VEL;
    target_action.twist.angular.z = data.ball1_angular;
    is_action = true;
}

void resetCallback(msgs::reset data){
    reset = data;
}

void stopperCallback(msgs::reset data){
    stopper = data.is_reset;
}

int main(int argc, char **argv){
    int RATE = 30;
    ros::init(argc, argv, "state_publisher");
    ros::NodeHandle node;
    ros::Rate rate(RATE);
    ros::Subscriber model_state_sub = node.subscribe("/gazebo/model_states", RATE, stateCallback);
    ros::Subscriber step_sub = node.subscribe("/step", RATE, actionCallback);
    ros::Subscriber reset_sub = node.subscribe("/reset", RATE, resetCallback);
    ros::Subscriber stopper_sub = node.subscribe("/stopper", RATE, stopperCallback);
    ros::Publisher set_model_state_pub = node.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", RATE);
    ros::Publisher obs_pub = node.advertise<msgs::observation>("/observation", RATE);
    ros::Publisher states_for_python = node.advertise<msgs::state>("/state", RATE);
    target_action.model_name = "agent";
    
    ros::service::call("/gazebo/reset_simulation", resetSimSrv);
    reset_env(set_model_state_pub, states_for_python);
    reset.is_reset = false;
    publish_obs(obs_pub, agent_pose, enemy_pose, agent_twist, enemy_twist, reward, done_indice);
    
    while(ros::ok()){

        if(is_action == true){
            take_action(set_model_state_pub, target_action);
            is_action = false;
        }
        
        done_status = is_done(agent_pose, enemy_pose);
        
        if(done_status == 1 || done_status == 0){
            if (done_status == 1){
                reward = SUCCESS_REWARD;
            }else{
                reward = REWARD;
            }
            done_indice = 1;
            publish_obs(obs_pub, agent_pose, enemy_pose, agent_twist, enemy_twist, reward, done_indice);
            // stopper = true;
            // int a = 0;
            // while(stopper == true){
            //     a += 1;
            // }

        }else{ //done statusu -1 ise burası geçerli oluyor.
            done_indice = 0;
            distance = calculate_distance(agent_pose, enemy_pose);
            reward = (1/distance)*REWARD;
            publish_obs(obs_pub, agent_pose, enemy_pose, agent_twist, enemy_twist, reward, done_indice);
        }
        
        if(reset.is_reset == true){
            ros::service::call("/gazebo/reset_simulation", resetSimSrv);
            reset_env(set_model_state_pub, states_for_python);
            reset.is_reset = false;
        }
    
        ros::spinOnce();
        rate.sleep();
    }   
}
