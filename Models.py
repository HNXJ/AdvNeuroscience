from matplotlib import pyplot as plt
import numpy as np
import warnings
import time


warnings.filterwarnings('ignore')


class SessionLog:
  def __init__(self):
    self.path = None
    self.maze = None
    self.error = 0
    return

  def get_path(self):
    return self.path

  def get_maze(self):
    return self.maze

  def get_error(self):
    return self.error

  def set_error(self, error):
    self.error = error

  def set_path(self, path):
    self.path = path

  def set_maze(self, maze):
    self.maze = maze

class WaterMaze:

  def __init__(self, mapsize=(17, 17), eps=0.1, noisyRun=False):
    self.maze = np.zeros(mapsize)
    self.maze += 0.5

    self.maze[:, 0] = -2
    self.maze[:, 16] = -2
    self.maze[0, :] = -2
    self.maze[16, :] = -2
    
    self.eps = eps
    self.maze[3, 4] = 1.5
    self.maze[9, 11] = -1.5
    self.currentpath = 0

    self.paths = None
    self.logs = []
    self.fig = None
    self.ax = None
    return

  @staticmethod
  def getDirection(p, epsl=0.0):
    ps = p
    for i in range(1, 4):
      ps[i] += ps[i-1]
    ps /= ps[3]
    q = np.random.rand()

    r = np.random.rand()
    if r < epsl:
      return np.random.randint(0, 4)

    if q < ps[0]:
      return 0
    elif q < ps[1]:
      return 1
    elif q < ps[2]:
      return 2
    elif q < ps[3]:
      return 3

  @staticmethod
  def getArgmax(p, epsl=0.0):
    q = np.where(p == np.max(p))
    if q[0].shape[0] < 1:
      return 1
    idx = np.random.randint(0, q[0].shape[0])
    idx = q[0][idx]
    r = np.random.rand()
    if r < epsl:
      return np.random.randint(0, 4)
    return idx

  def getErrLog(self):
    y = np.zeros(len(self.logs))
    for i in range(len(self.logs)):
      y[i] = self.logs[i].error
    return y

  def errPlotter(self, ax, title="Mean accuracy over sessions"):
    y = self.getErrLog()
    ax.plot(y)
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.set_xlabel("Sessions")
    ax.grid(True)
    return

  def animate(self, ax, title="Session heatmap"):
    
      ax.imshow(self.maze)
      ax.text(4, 3+0.3, "T")
      ax.text(11, 9+0.3, "X")
      ax.set_title(title)

      for i in range(1, 16):
        for j in range(1, 16):
          ax.text(i-0.3, j-0.1, "%.4f" % (float(self.maze[j, i])))

      return

  def plotpath(self, ax, title, paths):
      # ax.imshow(self.maze)
      ax.text(4, 3+0.3, "T")
      ax.text(11, 9+0.3, "X")
      ax.set_title(title)

      for i in range(paths.shape[1]):
        ax.text(paths[1, i]-0.4, paths[0, i]+0.9, "*", fontsize=48)

      return

  def runTrial(self, x, y, mode="argmax", epsl=0.0, lambd=1.0):
    
    if x > 15:
      x -= 1
    if x < 1:
      x += 1
    if y > 15:
      y -= 1
    if y < 1:
      y += 1

    if self.maze[x, y] <= -1.49:
      self.flag = 1
      return -1, -1
    
    if self.maze[x, y] >= 1.49:
      self.flag = -1
      return -1, -1
    
    self.paths = np.concatenate([self.paths, np.array([[x], [y]])], 1)
    
    p = np.array([self.maze[x-1, y], self.maze[x+1, y] ,self.maze[x, y-1] ,self.maze[x, y+1]])
    
    if x <= 1:
      p[0] = 0
    elif x >= 15:
      p[1] = 0
    if y <= 1:
      p[2] = 0
    elif y >= 15:
      p[3] = 0

    # print(x, y, p)
    if mode=="argmax":
      idx = self.getArgmax(p, epsl=epsl)
    elif mode=="probabilistic":
      idx = self.getDirection(p, epsl=epsl)
    
    if idx == 0:
      return x-1, y
    elif idx == 1:
      return x+1, y
    elif idx == 2:
      return x, y-1
    else:
      return x, y+1
   
  def runSession(self, epochs=100, maxlen=100, animate=False, mode="argmax", epsl=0.0, lambd=0.5, flag=0):

    cnt = 0.0
    for i in range(epochs):
      xt = np.random.randint(1, 16)
      yt = np.random.randint(1, 16)

      self.flag = flag
      self.paths = np.array([[0], [0]])

      while self.paths.shape[1] < maxlen and (not xt == -1):
        xt, yt = self.runTrial(xt, yt, mode, epsl=epsl)

      if self.flag:
        for i in range(self.paths.shape[1]):
          self.maze[self.paths[0, i], self.paths[1, i]] -= self.eps*self.flag*(i/self.paths.shape[1])*np.exp(i*np.log(lambd))

      self.maze[3, 4] = 0.0
      self.maze[9, 11] = 0.0
      self.maze[1:16, 1:16] /= np.max(np.abs(self.maze[1:16, 1:16]))
      self.maze[1:16, 1:16] -= np.min(self.maze[1:16, 1:16])
      self.maze[3, 4] = 1.5
      self.maze[9, 11] = -1.5   

      slog = SessionLog()

      if self.flag < 0:
        cnt = cnt*0.95 + 0.05 + 0.2*(1 - cnt)**2
      else:
        cnt = cnt*0.95

      slog.set_error(cnt)
      slog.set_path(self.paths)
      slog.set_maze(self.maze*1.0)
      self.logs.append(slog)

      if animate:
        time.sleep(0.500)
        self.animate()

  def runSessionTD(self, epochs=100, maxlen=100, animate=False, mode="argmax", epsl=0.0, gamma=1.0, lambds=1.0, flag=0):

    cnt = 0.0
    for i in range(epochs):
      xt = np.random.randint(1, 16)
      yt = np.random.randint(1, 16)

      self.flag = flag
      self.paths = np.array([[0], [0]])

      while self.paths.shape[1] < maxlen and (not xt == -1):
        xt, yt = self.runTrial(xt, yt, mode, epsl=epsl, lambd=lambds)
    
      if self.flag:
        for i in range(self.paths.shape[1]):
          if i:
            tdl = self.maze[self.paths[0, i-1], self.paths[1, i-1]]*lambds
          else:
            tdl = 0
          self.maze[self.paths[0, i], self.paths[1, i]] -= (self.eps*self.flag*(i/self.paths.shape[1])*np.exp(i*np.log(gamma) - tdl))

      self.maze[3, 4] = 0.0
      self.maze[9, 11] = 0.0
      self.maze[1:16, 1:16] /= np.max(np.abs(self.maze[1:16, 1:16]))
      self.maze[1:16, 1:16] -= np.min(self.maze[1:16, 1:16])
      self.maze[3, 4] = 1.5
      self.maze[9, 11] = -1.5   

      slog = SessionLog()
      if self.flag < 0:
        cnt = cnt*0.95 + 0.05 + 0.2*(1 - cnt)**2
      else:
        cnt = cnt*0.95
      
      slog.set_error(cnt)
      slog.set_path(self.paths)
      slog.set_maze(self.maze*1.0)
      self.logs.append(slog)

      if animate:
        time.sleep(0.500)
        self.animate()


class WaterMazeT:

  def __init__(self, mapsize=(17, 17), eps=0.1, noisyRun=False):
    self.maze = np.zeros(mapsize)
    self.maze += 0.5

    self.maze[:, 0] = -2
    self.maze[:, 16] = -2
    self.maze[0, :] = -2
    self.maze[16, :] = -2
    
    self.eps = eps
    self.maze[3, 4] = 1.5
    self.maze[5, 12] = 2.5
    self.maze[9, 11] = -1.5
    self.currentpath = 0

    self.paths = None
    self.logs = []
    self.fig = None
    self.ax = None
    return

  @staticmethod
  def getDirection(p, epsl=0.0):
    ps = p
    for i in range(1, 4):
      ps[i] += ps[i-1]
    ps /= ps[3]
    q = np.random.rand()

    r = np.random.rand()
    if r < epsl:
      return np.random.randint(0, 4)

    if q < ps[0]:
      return 0
    elif q < ps[1]:
      return 1
    elif q < ps[2]:
      return 2
    elif q < ps[3]:
      return 3

  @staticmethod
  def getArgmax(p, epsl=0.0):
    q = np.where(p == np.max(p))
    if q[0].shape[0] < 1:
      return 1
    idx = np.random.randint(0, q[0].shape[0])
    idx = q[0][idx]
    r = np.random.rand()
    if r < epsl:
      return np.random.randint(0, 4)
    return idx

  def getErrLog(self):
    y = np.zeros(len(self.logs))
    for i in range(len(self.logs)):
      y[i] = self.logs[i].error
    return y

  def errPlotter(self, ax, title="Mean accuracy over sessions"):
    y = self.getErrLog()
    ax.plot(y)
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.set_xlabel("Sessions")
    ax.grid(True)
    return

  def animate(self, ax, title="Session heatmap"):
    
      ax.imshow(self.maze)
      ax.text(4, 3+0.3, "T")
      ax.text(5, 12+0.3, "T")
      ax.text(11, 9+0.3, "X")
      ax.set_title(title)

      for i in range(1, 16):
        for j in range(1, 16):
          ax.text(i-0.3, j-0.1, "%.4f" % (float(self.maze[j, i])))

      return

  def plotpath(self, ax, title, paths):
      # ax.imshow(self.maze)
      ax.text(4, 3+0.3, "T")
      ax.text(5, 12+0.3, "T")
      ax.text(11, 9+0.3, "X")
      ax.set_title(title)

      for i in range(paths.shape[1]):
        ax.text(paths[1, i]-0.4, paths[0, i]+0.9, "*", fontsize=48)

      return

  def runTrial(self, x, y, mode="argmax", epsl=0.0, lambd=1.0):
    
    if x > 15:
      x -= 1
    if x < 1:
      x += 1
    if y > 15:
      y -= 1
    if y < 1:
      y += 1

    if self.maze[x, y] <= -1.49:
      self.flag = 1
      return -1, -1
    
    if self.maze[x, y] >= 1.49:
      self.flag = -1
      return -1, -1
    
    self.paths = np.concatenate([self.paths, np.array([[x], [y]])], 1)
    
    p = np.array([self.maze[x-1, y], self.maze[x+1, y] ,self.maze[x, y-1] ,self.maze[x, y+1]])
    
    if x <= 1:
      p[0] = 0
    elif x >= 15:
      p[1] = 0
    if y <= 1:
      p[2] = 0
    elif y >= 15:
      p[3] = 0

    # print(x, y, p)
    if mode=="argmax":
      idx = self.getArgmax(p, epsl=epsl)
    elif mode=="probabilistic":
      idx = self.getDirection(p, epsl=epsl)
    
    if idx == 0:
      return x-1, y
    elif idx == 1:
      return x+1, y
    elif idx == 2:
      return x, y-1
    else:
      return x, y+1
   
  def runSession(self, epochs=100, maxlen=100, animate=False, mode="argmax", epsl=0.0, lambd=0.5, flag=0):

    cnt = 0.0
    for i in range(epochs):
      xt = np.random.randint(1, 16)
      yt = np.random.randint(1, 16)

      self.flag = flag
      self.paths = np.array([[0], [0]])

      while self.paths.shape[1] < maxlen and (not xt == -1):
        xt, yt = self.runTrial(xt, yt, mode, epsl=epsl)

      if self.flag:
        for i in range(self.paths.shape[1]):
          self.maze[self.paths[0, i], self.paths[1, i]] -= self.eps*self.flag*(i/self.paths.shape[1])*np.exp(i*np.log(lambd))

      self.maze[3, 4] = 0.0
      self.maze[5, 12] = 0.0
      self.maze[9, 11] = 0.0

      self.maze[1:16, 1:16] /= np.max(np.abs(self.maze[1:16, 1:16]))
      self.maze[1:16, 1:16] -= np.min(self.maze[1:16, 1:16])
      
      self.maze[3, 4] = 1.5
      self.maze[5, 12] = 2.5
      self.maze[9, 11] = -1.5   

      slog = SessionLog()

      if self.flag < 0:
        cnt = cnt*0.95 + 0.05 + 0.2*(1 - cnt)**2
      else:
        cnt = cnt*0.95

      slog.set_error(cnt)
      slog.set_path(self.paths)
      slog.set_maze(self.maze*1.0)
      self.logs.append(slog)

      if animate:
        time.sleep(0.500)
        self.animate()

  def runSessionTD(self, epochs=100, maxlen=100, animate=False, mode="argmax", epsl=0.0, gamma=1.0, lambds=1.0, flag=0):

    cnt = 0.0
    for i in range(epochs):
      xt = np.random.randint(1, 16)
      yt = np.random.randint(1, 16)

      self.flag = flag
      self.paths = np.array([[0], [0]])

      while self.paths.shape[1] < maxlen and (not xt == -1):
        xt, yt = self.runTrial(xt, yt, mode, epsl=epsl, lambd=lambds)
    
      if self.flag:
        for i in range(self.paths.shape[1]):
          if i:
            tdl = self.maze[self.paths[0, i-1], self.paths[1, i-1]]*lambds
          else:
            tdl = 0
          self.maze[self.paths[0, i], self.paths[1, i]] -= (self.eps*self.flag*(i/self.paths.shape[1])*np.exp(i*np.log(gamma) - tdl))

      self.maze[3, 4] = 0.0
      self.maze[5, 12] = 0.0
      self.maze[9, 11] = 0.0

      self.maze[1:16, 1:16] /= np.max(np.abs(self.maze[1:16, 1:16]))
      self.maze[1:16, 1:16] -= np.min(self.maze[1:16, 1:16])
      
      self.maze[3, 4] = 1.5
      self.maze[5, 12] = 2.5
      self.maze[9, 11] = -1.5   

      slog = SessionLog()
      if self.flag < 0:
        cnt = cnt*0.95 + 0.05 + 0.2*(1 - cnt)**2
      else:
        cnt = cnt*0.95
      
      slog.set_error(cnt)
      slog.set_path(self.paths)
      slog.set_maze(self.maze*1.0)
      self.logs.append(slog)

      if animate:
        time.sleep(0.500)
        self.animate()
