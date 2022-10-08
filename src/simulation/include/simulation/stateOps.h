#include <iostream>
#include <thread>
#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Twist.h"
#include "gazebo_msgs/ModelStates.h"
#include "gazebo_msgs/SetModelState.h"
#include "msgs/state.h"
#include "msgs/action.h"
#include "msgs/reset.h"
#include "msgs/observation.h"



void publish_obs(ros::Publisher obs_pub, geometry_msgs::Pose agent_pose, geometry_msgs::Pose enemy_pose,
                 geometry_msgs::Twist agent_twist, geometry_msgs::Twist enemy_twist, float reward, int done_indice){
    msgs::observation obs;
    obs.agent_x = agent_pose.position.x;
    obs.agent_y = agent_pose.position.y;
    obs.agent_z = agent_pose.position.z;
    obs.enemy_x = enemy_pose.position.x;
    obs.enemy_y = enemy_pose.position.y;
    obs.enemy_z = enemy_pose.position.z;
    obs.agent_vx = agent_twist.linear.x;
    obs.agent_vy = agent_twist.linear.y;
    obs.agent_vz = agent_twist.linear.z;
    obs.agent_w = agent_twist.angular.z;
    obs.enemy_vx = enemy_twist.linear.x;
    obs.enemy_vy = enemy_twist.linear.y;
    obs.enemy_vz = enemy_twist.linear.z;
    obs.reward = reward;
    obs.done_indice = done_indice;
    obs_pub.publish(obs);
    }

//This function publishes state for python reinforcement env.
void publish_state(ros::Publisher publisher, gazebo_msgs::ModelState agent_state, gazebo_msgs::ModelState enemy_state){
    geometry_msgs::Pose agent_pose;
    geometry_msgs::Twist agent_twist;
    geometry_msgs::Pose enemy_pose;
    geometry_msgs::Twist enemy_twist;
    msgs::state msg;
    msg.ball1_x = agent_pose.position.x;
    msg.ball1_y = agent_pose.position.y;
    msg.ball1_z = agent_pose.position.z;
    msg.ball2_x = enemy_pose.position.x;
    msg.ball2_y = enemy_pose.position.y;
    msg.ball2_z = enemy_pose.position.z;
    msg.ball1_vx = agent_twist.linear.x;
    msg.ball1_vy = agent_twist.linear.y;
    msg.ball1_vz = agent_twist.linear.z;
    msg.ball1_w = agent_twist.angular.z;
    msg.ball2_vx = enemy_twist.linear.x;
    msg.ball2_vy = enemy_twist.linear.y;
    msg.ball2_vz = enemy_twist.linear.z;
    publisher.publish(msg);
    }


// This func. upgrades model states, other words, it publishes data to set_model_state topic.
void take_action(ros::Publisher publisher, gazebo_msgs::ModelState state){
    // state.pose.position.x += state.twist.linear.x/60;
    // state.pose.position.y += state.twist.linear.y/60;
    // state.pose.position.z += state.twist.linear.z/60;
    // state.twist.linear.x = 0;
    // state.twist.linear.y = 0;
    // state.twist.linear.z = 0;
    // state.twist.angular.z = 0;
    publisher.publish(state);
    }


void reset_env(ros::Publisher set_model_state_pub, ros::Publisher state_publisher){
    gazebo_msgs::ModelState agent_state = init_state("agent");
    gazebo_msgs::ModelState enemy_state = init_state("enemy");
    enemy_state.pose.position.z = 0.1;
    set_model_state_pub.publish(agent_state);
    set_model_state_pub.publish(enemy_state);
    }