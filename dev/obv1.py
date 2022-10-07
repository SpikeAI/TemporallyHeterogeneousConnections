import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

fig_width = 15
#phi = np.sqrt(5)/2 + 1/2

def surprise_1D(b, p):
    return -(b*np.log2(p) + (1-b)*np.log2(1-p))

def surprise(b, p):
    if p.ndim==1:
        return -(b*np.log2(p[None, :]) + (1-b)*np.log2(1-p[None, :])).mean(axis=1)
    if p.ndim==2:
        return -(b[:, :, None]*np.log2(p[None, :, :]) + (1-b[:, :, None])*np.log2(1-p[None, :, :])).mean(axis=1)
    
# a simple circular function to generate patterns
def von_mises(j, N, sigma, p1):
    p = np.exp( np.cos(2*np.pi* (np.linspace(0, 1, N, endpoint=False) -j/N)) / sigma**2)
    p /= p.mean()/p1
    return p

def stack(K, N, sigma, p1):
    p = np.zeros((N, K))
    for k in range(K):
        p[:, k] = von_mises(k*N/K, N, sigma, p1)
    return p

def generative_model(p, N_trials):
    N = p.size
    b = np.random.rand(N_trials, N) < p[None, :]
    return b

def get_confusion_matrix(p_true, p_hat, N_trials):
    N_true, N_hat = p_true.shape[-1], p_hat.shape[-1]
    confusion_matrix = np.zeros((N_true, N_hat))
    for i_test in range(N_true):
        b = generative_model(p_true[:, i_test], N_trials)
        # matching
        k_star = surprise(b, p_hat).argmin(axis=-1)
        for i_hyp in range(N_hat):
            confusion_matrix[i_test, i_hyp] = (k_star == i_hyp).mean()    
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width))

    cmap = ax.matshow(confusion_matrix, vmin=0, vmax=1)
    width, height = confusion_matrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate('%.2f' % confusion_matrix[x][y], xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=35, color='white' if confusion_matrix[x][y]<.5 else 'red')
    plt.colorbar(cmap)
    ax.set_ylabel('True class', fontsize=32)
    ax.set_xlabel('Predicted class', fontsize=32)
    return fig, ax

def evaluate(K, N, sigma, p1, N_epochs, eta, N_trials, noise=1):
    p_true = stack(K, N, sigma=sigma, p1=p1)
    if N_epochs==0:
        confusion_matrix = get_confusion_matrix(p_true, p_true, N_trials)
    else:
        p_hat = learn(p_true, N_epochs, eta, noise, N_trials, p1)
        confusion_matrix = get_confusion_matrix(p_true, p_hat, N_trials)
        
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    # from sklearn.metrics.cluster import adjusted_rand_score
    # adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
  
    return confusion_matrix.max(axis=0).mean()

def get_data(p_true, N_trials, do_warmup = True, do_shuffle=True):
    N, K = p_true.shape
    if do_warmup :
        b = generative_model(p_true.mean() * np.ones(N), N_trials//100)
    else :
        b = np.zeros((0, N))
    order = np.arange(K)
    if do_shuffle: np.random.shuffle(order)
    for k in order:
        b = np.vstack((b, generative_model(p_true[:,k], N_trials)))
    return b

def learn(p_true, N_epochs, eta, noise, N_trials, p1):
    N, K = p_true.shape
    for i_epoch in trange(N_epochs, desc='learning', leave=False):
        b = get_data(p_true, N_trials)
        if i_epoch==0:
            # init
            p_hat = np.ones((N, K))
            for i_test in range(K):
                p_hat[:, i_test] *= b.mean(axis=0)
                p_hat[:, i_test] *= 1 + noise*np.random.rand(N)
                p_hat[:, i_test] *= p1 / p_hat[:, i_test].mean()

        # matching
        k_star = surprise(b, p_hat).argmin(axis=-1)
        
        # hebbian learning
        eta_ = np.min((eta/(np.log(i_epoch+1)+1), 1.))
        for k in range(K):
            # frequency that pattern number *k* was selected
            proba_win = (k_star==k).sum() * K / N
            p_hat[:, k] *= 1 - eta_ * proba_win
            # observed pattern on the trials for which we selected number *k*
            p_new = b[k_star==k, :].mean(axis=0)
            # hebbian learning / moving average
            p_hat[:, k] += eta_ * proba_win * p_new
            
        # normalize    
        p_hat[:, i_test] *= p1  / p_hat[:, i_test].mean()      
    return p_hat

def learn2(p_hat, p_true, h, N_epochs, eta, alpha, N_trials, do_fr=False):
    N, K = p_true.shape
    proba_win = np.ones((N_epochs, K)) / K
    F = np.zeros(N_epochs)
    tq = trange(N_epochs, desc='F=>N/A< / learning', leave=True)
    for i_epoch in tq:
        proba_win[i_epoch, :] = np.zeros(K)
        # draw samples
        b = get_data(p_true, N_trials, do_warmup=False)
        T, N = b.shape

        # matching by finding for each trials the indices which correspond to the minimal surprise
        S = np.zeros((T, N_hyp))
        S[-1, :] = surprise(p1*np.ones(N)[None, :], p_true)

        for t in range(T-1):
            S[t, :] = (1-h) * S[t-1, :] + h * surprise(b[t, : ][None, :], p_hat)

        P = get_proba(S)
        
        if do_fr:
            # smoothed firing rate to use in learning
            f = np.zeros_like(b).astype(np.float)
            f[-1, : ] = b.mean()*np.ones(N)
            for t in range(T):
                f[t, : ] = (1-h) * f[t-1, : ] + h * b[t, :]

        #eta_ = eta/(np.log(i_epoch+1)+1) # scheduling weight decay
        eta_ = eta
        eta_T = 1/T
        for t in range(T):
            # TODO utiliser un ELBO pour pénaliser les réponses différentes d'un pic
            proba = P[t, :] * np.exp(- alpha * (proba_win[i_epoch-1, :] - 1/K) )
            k_star = proba.argmax()
            F[i_epoch] += S[t, k_star]/T # average surprise *knowing* our selection

            # frequency that pattern number *k* was selected
            proba_win[i_epoch, :] += eta_T * (np.arange(K) == k_star)

            # hebbian learning through a moving average
            #p_random = np.random.rand(N, N_hyp)
            #p_random *= b.mean()/p_random.mean()
            #p_hat = (1 - eta_) * p_hat + eta_ * p_random
            
            p_hat[:, k_star] *= (1 - eta_)
            if do_fr:
                p_hat[:, k_star] += eta_ * f[t, :]
            else:
                p_hat[:, k_star] += eta_ * b[t, :]

        #tq.set_description(f'F={F[i_epoch-1]:.1f}/ proba_win * K ={proba_win * K} / learning')
        tq.set_description(f'F={F[i_epoch-1]:.1f} / learning')
        tq.refresh() # to show immediately the update
      
    return p_hat, P, F, proba_win

def learn3(p_hat, p_true, h, N_epochs, eta, alpha, do_fr=False):
    proba_win = np.ones((N_epochs, K)) / K
    F = np.zeros(N_epochs)
    tq = trange(N_epochs, desc='F=>N/A< / learning', leave=True)
    for i_epoch in tq:
        proba_win[i_epoch, :] = np.zeros(K)
        # draw samples
        b = get_data(p_true, N_trials, do_warmup=False)
        T, N = b.shape

        # matching by finding for each trials the indices which correspond to the minimal surprise
        S = np.zeros((T, N_hyp))
        S[-1, :] = surprise(p1*np.ones(N)[None, :], p_true)

        for t in range(T-1):
            S[t, :] = (1-h) * S[t-1, :] + h * surprise(b[t, : ][None, :], p_hat)

        P = get_proba(S)
        
        if do_fr:
            # smoothed firing rate to use in learning
            f = np.zeros_like(b).astype(np.float)
            f[-1, : ] = b.mean()*np.ones(N)
            for t in range(T):
                f[t, : ] = (1-h) * f[t-1, : ] + h * b[t, :]

        #eta_ = eta/(np.log(i_epoch+1)+1) # scheduling weight decay
        eta_ = eta
        eta_T = 1/T
        for t in range(T):
            # TODO utiliser un ELBO pour pénaliser les réponses différentes d'un pic
            proba = P[t, :] * np.exp(- alpha * (proba_win[i_epoch-1, :] - 1/K) )
            k_star = proba.argmax()
            F[i_epoch] += S[t, k_star]/T # average surprise *knowing* our selection

            # frequency that pattern number *k* was selected
            proba_win[i_epoch, :] += eta_T * (np.arange(K) == k_star)

            # hebbian learning through a precision-weighted moving average          
            pi_hat = np.sum(p_hat[:, k_star] * (1 - p_hat[:, k_star]))
            p_hat[:, k_star] *= (1 - eta_/pi_hat)
            p_hat[:, k_star] += eta_/pi_hat * b[t, :]

        #tq.set_description(f'F={F[i_epoch-1]:.1f}/ proba_win * K ={proba_win * K} / learning')
        tq.set_description(f'F={F[i_epoch-1]:.1f} / learning')
        tq.refresh() # to show immediately the update
      
    return p_hat, P, F, proba_win

def get_proba(S):
    # softmax activation function
    P = np.exp(-np.log(2)*S) # S is in bits
    P /= P.sum(axis=1)[:, None]
    return P

class Data :
    def __init__(self, opt) :
        self.opt = opt
        self.d = vars(opt)
        
    def vm(self, theta_0, B_theta) :
        p = np.exp(np.cos(np.linspace(0, 2*np.pi, self.opt.N, endpoint=False) - theta_0) / B_theta**2)
        p /= p.mean()
        p *= self.opt.p_0
        return p
    
    def stack(self) :
        p = np.zeros((self.opt.N, self.opt.K))
        if self.opt.Bt_mode == "proportion" :
            # mode hierarchique  8 - 4 - 2
            raison = 1.5
            k, K, B_theta = 0, self.opt.K, self.opt.B_theta
            while k < self.opt.K:
                K = int(K/raison)
                print(f'K = {K}')
                for k_ in range(K) :
                    theta_0 = 2 * np.pi * (k + 1/2) / K
                    p[:,k] = self.vm(theta_0 = theta_0, B_theta = B_theta)
                    k += 1
                    print(f'k = {k}')
                B_theta = B_theta*2
        else:
            for k in range(self.opt.K) :
                theta_0 = 2 * np.pi * (k + 1/2) / self.opt.K

                # A single Btheta for all tuning curves
                if self.opt.Bt_mode == "single" :
                    B_theta = self.opt.B_theta
                # Random Bthetas between 50 and 150 % for each tuning curve
                elif self.opt.Bt_mode == "random" :
                    B_theta = self.opt.B_theta * np.random.uniform(.5, 1.5)
                #One third of Bthetas are bigger
                p[:,k] = self.vm(theta_0 = theta_0,
                                 B_theta = B_theta)
            
        return p

class OnlineBinaryDetection(Data):
    def __init__(self, opt):
        """
        Detection class
        """
        super().__init__(opt)

    def surprise(self, b, p):
        if p.ndim==1:
            return -(b*np.log2(p[None, :]) + (1-b)*np.log2(1-p[None, :])).sum(axis=1)
        if p.ndim==2:
            return -(b[:, :, None]*np.log2(p[None, :, :]) + (1-b[:, :, None])*np.log2(1-p[None, :, :])).sum(axis=1)
        
    def firing_rate(self, b):
        f = np.zeros_like(b*1)
        f[0, : ] = self.opt.p_0*np.ones(self.opt.N)
        T, N = b.shape

        for t in range(T):
            f[t, : ] = self.opt.h * b[t, : ] + (1-self.opt.h) * f[t-1, : ]
        return f
        
    def get_surprise(self, b, p):
        T, N = b.shape
        N_, K = p.shape
        assert(N == N_)
        assert(N == self.opt.N)
        assert(K == self.opt.K)
        
        S = np.zeros((T, self.opt.K))
        S[-1, : ] = self.surprise(self.opt.p_0*np.ones((1, self.opt.N)), p)

        for t in range(T):
            S[t, :] = (1-self.opt.h) * S[t-1, :] + self.opt.h * self.surprise(b[t, : ][None, :], p)
        return S

    def plot_signal(self, signal, xlabel='time (bin number)', ylabel='signal', alpha=.4):
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/phi))
        ax.plot(signal, alpha=alpha)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax
    
    def plot_surprise(self, S, alpha=.4):
        return self.plot_signal(S, ylabel='NLL', alpha=alpha)
    
    def get_proba(self, S):
        # softmax activation function
        P = np.exp(-np.log(2)*S) # S is in bits
        P /= P.sum(axis=1)[:, None]
        return P

    def plot_proba_out(self, P, alpha=.4):
        return self.plot_signal(P, ylabel='proba', alpha=alpha)
        
    def get_output(self, P, N_pop=100):
        T, K = P.shape
        b_out = np.zeros((T, 0))
        for k in range(K):
            b_out_ = np.random.rand(T, N_pop) < P[:, k][:, None]
            b_out = np.hstack((b_out, b_out_.astype(np.float)))
        return b_out
    
class AdaptiveBinaryClustering(OnlineBinaryDetection):
    def __init__(self, opt):
        """
        Learning class
        """
        super().__init__(opt)
        
    def prior(self, noise=1):
        p_hat = np.ones((self.opt.N, self.opt.K)) + noise*np.random.rand(self.opt.N, self.opt.K)
        p_hat *= self.opt.p_0/p_hat.mean()
        return p_hat

    def learn(self, p_true, do_fr=False):
        """
        Learn, knowing that the generative model is parameterized by `p_true`
        
        """
        
        proba_win = np.ones((self.opt.N_epochs, self.opt.K)) / self.opt.K
        F = np.zeros(self.opt.N_epochs)
        tq = trange(self.opt.N_epochs, desc='F=>N/A< / learning', leave=True)
        
        p_hat = self.prior()
        
        for i_epoch in tq:
            proba_win[i_epoch, :] = np.zeros(self.opt.K)
            # draw samples
            b = self.get_data(p_true, do_warmup=False)
            T, N = b.shape

            # matching by finding for each trials the indices which correspond to the minimal surprise
            S = self.get_surprise(b, p_hat)
            P = self.get_proba(S)

            if do_fr: f = self.firing_rate(b)
                
            #eta_ = self.opt.eta/(np.log(i_epoch+1)+1) # scheduling weight decay
            eta_ = self.opt.eta
            eta_T = 1/T
            
            for t in range(T):
                # TODO utiliser un ELBO pour pénaliser les réponses différentes d'un pic
                proba = P[t, :] * np.exp(- self.opt.alpha * (proba_win[i_epoch-1, :] - 1/self.opt.K) )
                k_star = proba.argmax()
                F[i_epoch] += S[t, k_star]/T # average surprise *knowing* our selection

                # frequency that pattern number *k* was selected
                proba_win[i_epoch, :] += eta_T * (np.arange(self.opt.K) == k_star)

                # hebbian learning 
                # through a precision-weighted moving average (TODO: make optional)
                pi_hat = np.sum(p_hat[:, k_star] * (1 - p_hat[:, k_star]))
                p_hat[:, k_star] *= (1 - eta_/pi_hat)
                p_hat[:, k_star] += eta_/pi_hat * b[t, :]

            #tq.set_description(f'F={F[i_epoch-1]:.1f}/ proba_win * K ={proba_win * K} / learning')
            tq.set_description(f'F={F[i_epoch-1]:.1f} / learning')
            tq.refresh() # to show immediately the update

        return p_hat, P, F, proba_win
   
    def plot_F(self, F, alpha=.4):
        fig, ax = self.plot_signal(F, xlabel='epochs', ylabel='free energy', alpha=alpha)
        #ax.plot([0, N_epochs], [F_true, F_true], '--', alpha=.4, lw=2)
        return fig, ax
    
    def plot_P_win(self, F, alpha=.4):
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/phi))
        ax.pcolormesh(np.arange(self.opt.N_epochs), np.arange(self.opt.K), proba_win.T, alpha=.9)
        ax.set_ylabel('Pr_win')
        ax.set_xlabel('epochs')
        return fig, ax

    def plot_p_hat(self, p_true, p, ms='-', alpha=.4):
        fig, ax = self.plot_proba(p_true, ms='--', alpha=alpha)
        for k in range(self.opt.K):
            ax.step(np.arange(self.opt.N), p[:, k], ms, alpha=alpha)
        return fig, ax