import numpy as np
import seaborn as sns
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 5
        num_listeners = 4
        num_speakers = 4
        num_landmarks = 6
        world.landmark_colors = np.array(
            sns.color_palette(n_colors=num_landmarks))
        world.listeners = []
        for li in range(num_listeners):
            agent = Agent()
            agent.i = li
            agent.name = 'agent %i' % agent.i
            agent.listener = True
            agent.collide = True
            agent.size = 0.075
            agent.silent = True
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.listeners.append(agent)
        world.speakers = []
        for si in range(num_speakers):
            agent = Agent()
            agent.i = si + num_listeners
            agent.name = 'agent %i' % agent.i
            agent.listener = False
            agent.collide = False
            agent.size = 0.075
            agent.movable = False
            agent.accel = 1.5
            agent.initial_mass = 1.0
            agent.max_speed = 1.0
            world.speakers.append(agent)
        world.agents = world.listeners + world.speakers
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.i = i + num_listeners + num_speakers
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
            landmark.color = world.landmark_colors[i]
        # make initial conditions
        self.reset_world(world)
        self.reset_cached_rewards()
        return world

    def reset_cached_rewards(self):
        self.pair_rewards = None

    def post_step(self, world):
        self.reset_cached_rewards()

    def reset_world(self, world):
        listen_inds = list(range(len(world.listeners)))
        np.random.shuffle(listen_inds)  # randomize which listener each episode
        landmark_inds = list(range(len(world.landmarks)))
        np.random.shuffle(landmark_inds)
        for i, speaker in enumerate(world.speakers):
            li = listen_inds[i]
            la_ind = landmark_inds[i]
            speaker.listen_ind = li
            speaker.goal_a = world.listeners[li]
            speaker.goal_b = world.landmarks[la_ind]
            # speaker.goal_b = np.random.choice(world.landmarks)
            speaker.color = np.array([0.25, 0.25, 0.25])
            world.listeners[li].color = speaker.goal_b.color + np.array([0.25, 0.25, 0.25])
            world.listeners[li].speak_ind = i

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.reset_cached_rewards()

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, world)

    def calc_rewards(self, world):
        rews = []
        for speaker in world.speakers:
            dist = np.sqrt(np.sum(np.square(speaker.goal_a.state.p_pos -
                                            speaker.goal_b.state.p_pos))) - (speaker.goal_a.size + speaker.goal_b.size)
            # rew = -dist
            rew = dist
            # print(rew)
            # if dist < (speaker.goal_a.size + speaker.goal_b.size) * 1.5:
            # rew += 10.
            rews.append(rew)

        # if any listener is colliding with any other listener, there will be a fixed penalty
        seed = 0
        for listener_a in world.listeners:
            for listener_b in world.listeners:
                if listener_a.collide and listener_a != listener_b and self.is_collision(listener_a, listener_b):
                    seed = 1
        if seed == 1:
            rews_final = sum(rews) + 1000
        else:
            rews_final = sum(rews)

        return rews_final

    def reward(self, agent, world):
        if self.pair_rewards is None:
            self.pair_rewards = self.calc_rewards(world)

        return self.pair_rewards

    def observation(self, agent, world):
        if agent.listener:
            obs = []
            # give listener index of their speaker
            obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # give listener communication from its speaker
            obs += [world.speakers[agent.speak_ind].state.c]
            # give listener its own position/velocity,
            obs += [agent.state.p_pos, agent.state.p_vel]

            # obs += [world.speakers[agent.speak_ind].state.c]
            # # # give listener index of their speaker
            # # obs += [agent.speak_ind == np.arange(len(world.speakers))]
            # # # give listener all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give listener its own velocity
            # obs += [agent.state.p_vel]
            # # give listener locations of all agents
            # # obs += [a.state.p_pos for a in world.agents]
            # # give listener locations of all landmarks
            # obs += [l.state.p_pos for l in world.landmarks]
            return np.concatenate(obs)
        else:  # speaker => Grey Tower => immovable
            obs = []
            # give speaker index of their listener
            obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # speaker gets position of listener and goal
            obs += [agent.goal_a.state.p_pos, agent.goal_b.state.p_pos]

            # # give speaker index of their listener
            # # obs += [agent.listen_ind == np.arange(len(world.listeners))]
            # # # give speaker all communications
            # # obs += [speaker.state.c for speaker in world.speakers]
            # # give speaker their goal color
            # obs += [agent.goal_b.color]
            # # give speaker their listener's position
            # obs += [agent.goal_a.state.p_pos]
            #
            # obs += [speaker.state.c for speaker in world.speakers]
            return np.concatenate(obs)
