import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.current_traj_return = np.zeros((args.batch_size, 1))
        self.initial_s = np.zeros((args.batch_size, args.state_dim))
        self.device = args.device
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done, current_traj_return = None, initial_s = None):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        if not current_traj_return is None:
            self.current_traj_return[self.count] = current_traj_return
        if not initial_s is None:
            self.initial_s[self.count] = initial_s
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.device)
        a = torch.tensor(self.a, dtype=torch.float).to(self.device)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.device)
        r = torch.tensor(self.r, dtype=torch.float).to(self.device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.device)
        done = torch.tensor(self.done, dtype=torch.float).to(self.device)
        current_traj_return = torch.tensor(self.current_traj_return, dtype=torch.float).to(self.device)
        initial_s = torch.tensor(self.initial_s, dtype=torch.float).to(self.device)
        return s, a, a_logprob, r, s_, dw, done, current_traj_return, initial_s
