from models import ActorCritic, ReplayMemory, Updater,gae_calculator,  normalize, visualize
from env import Environment
from msgs.msg import observation, reset
from msgs.msg import action as action_msg

from collections import deque
import torch
import rospy
import time
import numpy as np
import sys

observation_msg = observation()

def observation_callback(data):
    global observation_msg
    observation_msg = data

def exit_callback(data):
    if data.is_reset == True:
        sys.exit()

def main():
    global observation_msg
    rospy.Subscriber("/observation", observation, observation_callback)
    rospy.Subscriber("/exit",reset, exit_callback)
    action_publisher = rospy.Publisher("/step", action_msg, queue_size=10)
    reset_publisher = rospy.Publisher("/reset", reset, queue_size=10)
    stopper_publisher = rospy.Publisher("/stopper", reset, queue_size=10)
    rospy.init_node("reinforcement_ops")
    
    RATE = 30
    MODE = "test"
    MODEL_SAVE_FILE = "models"
    TEST_WEIGHT_FILE = "reward_980.pth"
    MEMORY_CAPACITY = 512*15
    BATCH_SIZE = 256
    EPOCHS = 5000
    MAX_STEP = 512*3

    AVG_EPOCH_MEMORY = 10
    MODEL_TRAINING_EPOCHS = 2

    HIDDEN_SIZE_1 = 128
    HIDDEN_SIZE_2 = 128
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_PARAM = 0.2
    CRITIC_DISCOUNT = 0.5
    ENTROPY_BETA = 0.001
    LEARNING_RATE = 0.001

    env = Environment(action_publisher,reset_publisher,stopper_publisher)
    agent = ActorCritic(env.observation_space.shape[0],env.action_space.shape[0],HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    # agent.load_state_dict(torch.load(f"./{MODEL_SAVE_FILE}/reward_905.pth"))
    memory = ReplayMemory(BATCH_SIZE,MEMORY_CAPACITY)
    model_updater = Updater(agent,CLIP_PARAM,CRITIC_DISCOUNT,ENTROPY_BETA,learning_rate=LEARNING_RATE)

    best_avg_reward = -300
    epoch_memory = deque(maxlen=AVG_EPOCH_MEMORY)

    if MODE == "train":
        total_reward_file = open(f"./{MODEL_SAVE_FILE}/rewards_data.txt","a")
        for epoch in range(1950,EPOCHS):
            total_reward = 0
            env.reset()
            time.sleep(1)
            state, _, _= env.observe(observation_msg)
            for i in range(MAX_STEP):
                tensor = torch.tensor(np.array(state), dtype=torch.float)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                env.step(torch.squeeze(action).tolist())
                next_state, reward, done = env.observe(observation_msg)
                total_reward += reward
                memory.add_data(state,action.detach().numpy(),log_prob.detach().numpy(),reward,value.detach().numpy(),int(done))
                state = next_state
                time.sleep(1/RATE)
                
                if len(memory.memory) == MEMORY_CAPACITY:
                    break
                
                if done == 1:
                    break

            epoch_memory.append(total_reward)
            avg_reward = int(np.array(epoch_memory).mean())
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.state_dict(),f"./{MODEL_SAVE_FILE}/reward_{int(best_avg_reward)}.pth")
                print("Checkpoint saved successfully...", "AVG_REWARD:", best_avg_reward)

            print(f"EPOCH_NO:{epoch}, REWARD:{int(total_reward)}, BEST_REWARD:{avg_reward}, BEST_AVERAGE_REWARD:{best_avg_reward}")
            total_reward_file.write(f"{epoch} {total_reward}\n")
            total_reward_file.flush()

            if len(memory) >= MEMORY_CAPACITY:
                next_state_tensor = torch.tensor([next_state],dtype=torch.float)
                _, next_value = agent(next_state_tensor)
                states, actions, log_probs, rewards, values, dones = memory.get_memory()
                returns = gae_calculator(next_value.detach().numpy(),rewards,dones,values[:,0].tolist(), gamma = GAMMA, lam=GAE_LAMBDA)
                batches = memory.generate_batches(states, actions, log_probs, returns,values)
                for _ in range(MODEL_TRAINING_EPOCHS):
                    for batch in batches:
                        batch = np.array(batch,dtype=object)
                        states, actions, log_probs, returns, values = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                        returns_tensor = torch.tensor(np.vstack(returns),dtype=torch.float).detach()
                        log_probs_tensor = torch.tensor(np.vstack(log_probs), dtype=torch.float).detach()
                        values_tensor = torch.tensor(np.vstack(values), dtype=torch.float).detach()
                        states_tensor = torch.tensor(np.vstack(states), dtype=torch.float)
                        actions_tensor = torch.tensor(np.vstack(actions), dtype=torch.float)

                        advantage = returns_tensor - values_tensor
                        advantage = normalize(advantage)
                        model_updater.update(states_tensor,actions_tensor, log_probs_tensor, returns_tensor, advantage)
                memory.memory.clear()

        visualize(f"./{MODEL_SAVE_FILE}/rewards_data.txt")

    elif MODE == "test":
        agent.load_state_dict(torch.load(f"./{MODEL_SAVE_FILE}/{TEST_WEIGHT_FILE}"))
        
        for _ in range(10):
            total_reward = 0
            env.reset()
            time.sleep(1)
            state, _, _= env.observe(observation_msg)
            for i in range(MAX_STEP):
                tensor = torch.tensor(np.array(state), dtype=torch.float)
                dist, value = agent(tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                env.step(torch.squeeze(action).tolist())
                next_state, reward, done = env.observe(observation_msg)
                total_reward += reward
                state = next_state
                time.sleep(1/RATE)
                if done == 1:
                    break
            print(f"REWARD:{total_reward}")

if __name__ == "__main__":
    try:
        main()

    except rospy.ROSInterruptException:
        pass
