
import numpy as np
from models.agent import Agent
from models.network_agent import NetworkAgent
from utils.noise_process import OrnsteinUhlenbeckProcess
import random
import traceback
import os
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Activation, Lambda, Reshape,MultiHeadAttention
import tensorflow.keras.backend as K


class FuzzyLight1(object):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        self.intersection_id = intersection_id
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        
        # ===== check num actions == num phases ============
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])
        # self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.num_action_dur = len(dic_traffic_env_conf["ACTION_DURATION"])
        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.num_intersections = dic_traffic_env_conf["NUM_INTERSECTIONS"]

        self.cyclicInd = [[0] * self.num_phases for _ in range(self.num_intersections)]
        self.cyclicInd2 = [0] * self.num_intersections

        self.Xs, self.Y = None, None

        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        self.max_lane = dic_traffic_env_conf["MAX_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]
        self.low = 1
        self.high = 40
        self.tau = 0.1
        self.critic_lr = 2e-3
        self.actor_lr = 1e-5
        self.critic_optimizer = Adam(self.critic_lr)
        self.actor_optimizer = Adam(self.actor_lr)
        self.discount_factor = 0.8
        self.std_dev = 2
        self.pretrained_path = dic_traffic_env_conf["PRETRAINED_PATH"]
        self.ou_noise = OrnsteinUhlenbeckProcess(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        if cnt_round == 0:

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                self.build_network()    
                self.load_network("round_0_inter_{0}".format(intersection_id))
            else:
                self.build_network()       
                if os.path.exists(self.pretrained_path):
                    self.load_network_transfer("pretrained_",  self.pretrained_path)        
        else:
            try:
                self.build_network()               
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

        # decay the epsilon
        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    

    @staticmethod
    def build_memory():
        return []

    # def get_lane_embedding(self, ins0, ins1):
    #     feat1 = Reshape((self.max_lane, 4, 1))(ins0)
    #     feat1 = Dense(4, activation="sigmoid")(feat1)
    #     feat1 = Reshape((self.max_lane, 16))(feat1)
    #     lane_feats_s = tf.split(feat1, self.max_lane, axis=1)
    #     MHA1 = MultiHeadAttention(8, 8, attention_axes=1)
    #     Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
    #     phase_feats_map_2 = []
    #     for i in range(self.num_phases):
    #         tmp_feat_1 = tf.concat([lane_feats_s[idx] for idx in self.phase_map[i]], axis=1)
    #         phase_feats_map_2.append(tmp_feat_1)
    #     # [batch, num_phase, dim]
    #     phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
    #     tmp_feat_2 = MHA1(phase_feat_all, phase_feat_all)
    #     tmp_feat_3 = Mean1(tmp_feat_2)
    
    #     selected_phase_feat = Reshape((16, ))(tmp_feat_3)
    #     hidden = Dense(16, activation="relu")(selected_phase_feat)
    #     hidden = Dense(16, activation="relu")(hidden)
    #     return hidden


    def get_lane_embedding(self, ins0, ins1):
        feat1 = Reshape((self.max_lane, 4, 1))(ins0)
        feat1 = Dense(4, activation="sigmoid")(feat1)
        feat1 = Reshape((self.max_lane, 16))(feat1)
        lane_feats_s = tf.split(feat1, self.max_lane, axis=1)
        MHA1 = MultiHeadAttention(4, 8, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        phase_feats_map_2 = []
        for i in range(self.num_phases):
            tmp_feat_1 = tf.concat([lane_feats_s[idx] for idx in self.phase_map[i]], axis=1)
            #tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_1)
            phase_feats_map_2.append(tmp_feat_3)
        # [batch, num_phase, dim]
        phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
        selected_phase_feat = Lambda(lambda x: tf.matmul(x[0], x[1]))([ins1, phase_feat_all])
        selected_phase_feat = Reshape((16, ))(selected_phase_feat)
        hidden = Dense(16, activation="relu")(selected_phase_feat)
        hidden = Dense(16, activation="relu")(hidden)
        return hidden

    

    def build_actor(self):
        ins0 = Input(shape=(self.max_lane*4, ))
        ins1 = Input(shape=(1, self.num_phases))
        hidden = self.get_lane_embedding(ins0, ins1)
        out =  Dense(256, activation="relu")(hidden)
        out =  Dense(256, activation="relu")(out)
        outputs = Dense(1, activation="sigmoid")(out)
        outputs = outputs * self.high
        return Model([ins0,ins1],outputs)

    def build_critic(self):
        # State as input
        ins0 = Input(shape=(self.max_lane*4, ))
        ins1 = Input(shape=(1, self.num_phases))
        hidden = self.get_lane_embedding(ins0, ins1)
        state_out = Dense(16, activation="relu")(hidden)
        state_out = Dense(32, activation="relu")(state_out)
        # Action as input
        action_input = Input(shape=(1))
        action_out = Dense(32, activation="relu")(action_input)
        # Both are passed through seperate layer before concatenating
        concat = Concatenate()([state_out, action_out])
        out = Dense(256, activation="relu")(concat)
        out = Dense(256, activation="relu")(out)
        outputs = Dense(1)(out)
        return Model([ins0, ins1, action_input],outputs)

    def build_network(self): 
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_target = self.build_actor()
        self.critic_target = self.build_critic()

    def predict(self, state, noise_object, isnoise=True):
        sampled_actions = tf.squeeze(self.actor(state))
       
        # Adding noise to action
        if isnoise:
            sampled_actions = sampled_actions.numpy()
            if sampled_actions.size == 1:
                noise = noise_object.generate()
                sampled_actions += noise
            else:
                for i in range(sampled_actions.size):
                    noise = noise_object.generate()
                    sampled_actions[i] += noise
        print(sampled_actions)
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.low, self.high).astype(int).tolist()
        return legal_action

    def choose_action(self, states, list_need, noise=True):
            phase = []
            phase2 = []
            phase_feat = []
            for s in states:
                feat1 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][1]]
                feat0 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][0]]
                tmp_idx = self.phase_control_policy(feat0)
                phase.append([[tmp_idx]])
                phase2.append(tmp_idx)
                phase_feat.append(feat1)
            phase_feat2, phase_idx = np.array(phase_feat), np.array(phase)
            phase_matrix = self.phase_index2matrix(phase_idx)
            action = self.predict([phase_feat2,phase_matrix], self.ou_noise, noise)
            # for a in action:
            #     if a > 10:
            #         print('predict time > 10' + str(a))
            # if noise:
            #     self.t -= 1
            return phase2, action

    def phase_index2matrix(self, phase_index):
        # [batch, 1] -> [batch, 1, num_phase]
        lab = to_categorical(phase_index, num_classes=self.num_phases)
        return lab
    
    def train_network(self):
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Xs))
        for _ in range(epochs):
            sample_batch = random.sample(self.Xs, batch_size)
            state, phase, n_state = [],[],[]
            for item in sample_batch:
                state.append(item[0])
                phase.append(item[6])
                n_state.append(item[3])
            state = np.array(state)
            phase = np.array(phase)
            n_state = np.array(n_state)
            state_batch = [state, phase]
            next_state_batch = [n_state, phase]
            action_batch = tf.convert_to_tensor([[i[2]] for i in sample_batch])
            reward_batch = tf.convert_to_tensor([[i[4]] for i in sample_batch], dtype=float)     
            with tf.GradientTape() as tape:
                # actor预测下一步的v值
                target_actions = self.actor_target(next_state_batch, training=True)
                # 奖励+critic_target预测下一步q值  bellman公式
                y = tf.add(reward_batch ,self.discount_factor * self.critic_target(
                    [next_state_batch, target_actions], training=True
                ))
                # critic预测当前q值
                critic_value = self.critic([state_batch, action_batch], training=True)
                # critic_target预测q值与critic预测q值距离 奖励r的远近
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                tf.print('critic loss:', critic_loss)
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables)
            )

            with tf.GradientTape() as tape:
                # 当前actor预测v值
                actions = self.actor(state_batch, training=True)
                # 当前critic预测q值
                critic_value = self.critic([state_batch, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                # 做梯度上升
                actor_loss = -tf.math.reduce_mean(critic_value)
                tf.print('actor loss:', actor_loss)
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables)
            )
            self.update_target(self.actor_target.variables, self.actor.variables, self.tau)
            self.update_target(self.critic_target.variables, self.critic.variables, self.tau)

    def phase_control_policy(self, feat0):
        if self.num_lane == 8:
            feat10 = feat0[1] + feat0[3]
            feat20 = feat0[5] + feat0[7]
            feat30 = feat0[0] + feat0[2]
            feat40 = feat0[4] + feat0[6]
        elif self.num_lane == 10:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[9]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[8]
        elif self.num_lane == 12:
            feat10 = feat0[1] + feat0[4]
            feat20 = feat0[7] + feat0[10]
            feat30 = feat0[0] + feat0[3]
            feat40 = feat0[6] + feat0[9]
        elif self.num_lane == 16:
            feat10 = feat0[1] + feat0[2] + feat0[5] + feat0[6]
            feat20 = feat0[9] + feat0[10] + feat0[13] + feat0[14]
            feat30 = feat0[0] + feat0[4]
            feat40 = feat0[8] + feat0[12]
        idx = np.argmax([feat10, feat20, feat30, feat40])
        return idx

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        #  used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        #  use average reward

        for i in range(len(sample_slice)):
            state, action1, action2, next_state, reward, _ = sample_slice[i]
            phase_matrix = self.phase_index2matrix(np.array([action1]))
            
            sample_slice[i].append(phase_matrix)
        self.Xs = sample_slice

    def save_network(self, file_name):
        self.actor.save_weights(file_name + '_actor.h5')
        self.actor_target.save_weights(file_name +'_actor_t.h5')
        self.critic.save_weights(file_name + '_critic.h5')
        self.critic_target.save_weights(file_name +'_critic_t.h5')

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.actor.load_weights(file_path + '/' +file_name + '_actor.h5')
        self.actor_target.load_weights(file_path + '/' +file_name +'_actor_t.h5')
        self.critic.load_weights(file_path + '/' +file_name + '_critic.h5')
        self.critic_target.load_weights(file_path + '/' +file_name +'_critic_t.h5')
        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path):
        self.actor.load_weights(file_path + '/' +file_name + '_actor.h5')
        self.actor_target.load_weights(file_path + '/' +file_name +'_actor_t.h5')
        self.critic.load_weights(file_path + '/' +file_name + '_critic.h5')
        self.critic_target.load_weights(file_path + '/' +file_name +'_critic_t.h5')
        print("succeed in loading pretrained model %s" % file_name)