import random
from enum import Enum
import numpy as np

D = 10  #size of Array
alpha = 0.05
K = 1

class Cell:

    def __init__(self, x, y):
        self.xCor = x
        self.yCor = y
        self.status = 0 # 0: Closed 1: Open
        self.leakStatus = 1 # 0: No Leak 1: Maybe 2:Leak
        self.prob = 0

    def __str__(self):
        return f'X:{self.xCor} Y:{self.yCor} status:{self.status},{self.leakStatus} prob:{self.prob}\n'

    def __repr__(self):
        return f'X:{self.xCor} Y:{self.yCor} status:{self.status},{self.leakStatus} prob:{self.prob}\n'

    def printState(self, botLocation, leakLocation, secondLeak):
      if botLocation is not None and botLocation.xCor == self.xCor and botLocation.yCor == self.yCor:
        print ('■', end= '')
      elif leakLocation is not None and leakLocation.xCor == self.xCor and leakLocation.yCor == self.yCor:
        print ('@', end= '')
      elif secondLeak is not None and secondLeak.xCor == self.xCor and secondLeak.yCor == self.yCor:
        print ('@', end= '')
      else:
        if self.status == 0:
          print ('∅', end='')
        else:
          if self.leakStatus == 0:
            print ('O', end='')
          else :
            print ('L', end='')
      print(f' {self.prob:.2f}   ', end='')

def printArray(ship, botLocation, leakLocation, secondLeak):
  for j in range(0,D):
    for i in range(0,D):
      ship[i][j].printState(botLocation, leakLocation,secondLeak)
    print ('')

def checkifOneOpen(ship, x, y):
    i = 0
    if (x > 0 and ship[x-1][y].status == 1):
        i += 1
    if (x < D-1 and ship[x+1][y].status == 1):
        i += 1
    if (y > 0 and ship[x][y-1].status == 1):
        i += 1
    if (y < D-1 and ship[x][y+1].status == 1):
        i += 1
    if (i != 1):
        return False
    else:
        return True

def openNextCell(ship):
    checkArr = np.zeros((D, D), dtype = int)
    found = False
    while (found == False):
        nextX = random.randint(0,D-1)
        nextY = random.randint(0,D-1)
        checkArr[nextX][nextY] = 1
        if (ship[nextX][nextY].status == 0 and checkifOneOpen(ship, nextX, nextY) == True):
            #print(f'Opening at {nextX}, {nextY}')
            ship[nextX][nextY].status = 1
            found = True
        if np.any(checkArr == 0) != True:
            break
    return found

def placeBot(ship):
    checkArr = np.zeros((D, D), dtype = int)
    found=False
    while (found == False):
        nextX = random.randint(0,D-1)
        nextY = random.randint(0,D-1)
        if (checkArr[nextX][nextY] == 0):
            if ship[nextX][nextY].status == 1:
                selected=ship[nextX][nextY]
                found=True
            else:
                checkArr[nextX][nextY] = 1
                if np.any(checkArr == 0) != True:
                    break
    return selected

def placeLeak(ship, botLocation):
    checkArr = np.zeros((D, D), dtype = int)
    found=False
    while (found == False):
        nextX = random.randint(0,D-1)
        nextY = random.randint(0,D-1)
        if (checkArr[nextX][nextY] == 0):
          cell = ship[nextX][nextY]
          if cell.status == 1 and botLocation.xCor != nextX and botLocation.yCor != nextY:
            selected=cell
            found=True
          else:
            checkArr[nextX][nextY] = 1
            if np.any(checkArr == 0) != True:
              break
    return selected

def placeSecondLeak(ship, botLocation, leakLocation):
    checkArr = np.zeros((D, D), dtype = int)
    found=False
    while (found == False):
        nextX = random.randint(0,D-1)
        nextY = random.randint(0,D-1)
        if (checkArr[nextX][nextY] == 0):
          cell = ship[nextX][nextY]
          if cell.status == 1 and botLocation.xCor != nextX and botLocation.yCor != nextY and leakLocation.xCor != nextX and leakLocation.yCor != nextY:
            selected=cell
            found=True
          else:
            checkArr[nextX][nextY] = 1
            if np.any(checkArr == 0) != True:
              break
    return selected

# for x values, 0 is the top line, 3 is the bottom line - for y values, 0 is the value all the way on the left, 3 is the value all the way on the right
def checkDeadEnd(ship, x, y):
  i = 0
  if (x > 0 and ship[x-1][y].status == 1):
        i += 1
  if (x < D-1 and ship[x+1][y].status == 1):
        i += 1
  if (y > 0 and ship[x][y-1].status == 1):
        i += 1
  if (y < D-1 and ship[x][y+1].status == 1):
        i += 1
  if (ship[x][y].status) == 1 and i == 1:
    return True;
  else:
    return False;

def buildShipLayout():#Build Ship Layout
  #Setup the Ship Layout
  shipLayout = [[Cell(j,i) for i in range(D)] for j in range(D)]

  startX = random.randint(0, D-1)
  startY = random.randint(0, D-1)

  shipLayout[startX][startY].status = 1

  i = 0
  while openNextCell(shipLayout):
    i +=1

  deadCellCt = 0
  deadCells = []
  for i in range(D):
    for j in range(D):
      if checkDeadEnd(shipLayout, i, j) == True:
        #print(f'Cell {i},{j} is a dead end')
        deadCells.append(shipLayout[i][j])
        deadCellCt += 1

  if deadCellCt > 0 :
    if deadCellCt % 1 == 1:
      deadCellCt = int(deadCellCt/2 + 1)
    else:
      if deadCellCt % 1 == 0:
        deadCellCt = int(deadCellCt/2)

  #print (f'Flipping {deadCellCt} Cells')
  for i in range(deadCellCt):
    found = False
    #check up
    cell = deadCells[i]
    checkArr = [0,0,0,0]
    while (found == False):
        #get a random number 0 through 4
        # 0 is left 1 is right 2 is up 3 is down
        j = random.randint(0, 3)
        if (j == 0 and cell.xCor > 0 and shipLayout[cell.yCor][cell.xCor-1].status == 0):
            shipLayout[cell.yCor][cell.xCor-1].status = 1
            #print(f'opening {cell.yCor}, {cell.xCor - 1}')
            found = True
        if (found == False and j == 1 and cell.xCor < D-1 and shipLayout[cell.yCor][cell.xCor+1].status == 0):
            shipLayout[cell.yCor][cell.xCor+1].status = 1
            #print(f'opening {cell.yCor}, {cell.xCor + 1}')
            found = True
        if (found==False and j == 2 and cell.yCor > 0 and shipLayout[cell.yCor-1][cell.xCor].status == 0):
            shipLayout[cell.yCor-1][cell.xCor].status = 1
            #print(f'opening {cell.yCor - 1}, {cell.xCor}')
            found = True
        if (found==False and j == 3 and cell.yCor < D-1 and shipLayout[cell.yCor+1][cell.xCor].status == 0):
            shipLayout[cell.yCor+1][cell.xCor].status = 1
            #print(f'opening {cell.yCor+1}, {cell.xCor}')
            found = True
        checkArr[j] = 1
        if checkArr[0] == 1 and checkArr[1] == 1 and checkArr[2] == 1 and checkArr[3] == 1 :
            break
  return shipLayout

class QItem:
    def __init__(self, node, dist, oldpath):
        self.node = node
        self.dist = dist
        self.path = oldpath[:]

    def __repr__(self):
        return f"QItem({self.node}, {self.dist})"

# checking where move is valid or not
def isValid(n, grid, visited):
    flag = False
    if ((n.status == 1) and (visited[n.xCor][n.yCor] == False)):
        flag = True
    print(f'{n} is {flag} as {n.status}')
    return flag

def printVisited(arr):
  for j in range(D):
    for i in range(D):
      print(f'{arr[i][j]} ', end = '')
    print('')

def shortestPath(arr, start, goal):
  # To maintain location visit status
  if (start == goal):
    return []
  visited = [[False for i in range(D)] for j in range(D)]
  # applying BFS on matrix cells starting from source
  queue = []
  source = QItem(start, 0,[])
  queue.append(source)
  visited[start.xCor][start.yCor] = True
  while len(queue) != 0:
    source = queue.pop(0)
    #print(f'{source}')
    # Destination found;
    this = source.node
    if (this == goal):
        source.path.pop(0)
        source.path.append(this)
        return source.path
    # moving left
    if this.xCor > 0 :
      n = arr[this.xCor - 1][this.yCor]
      if n.status == 1 and visited[n.xCor][n.yCor] == False:
        q = QItem(n, source.dist + 1, source.path)
        q.path.append(this)
        queue.append(q)
        visited[n.xCor][n.yCor] = True
    if this.xCor < D-1 :
      n = arr[this.xCor + 1][this.yCor]
      if n.status == 1 and visited[n.xCor][n.yCor] == False:
        q = QItem(n, source.dist + 1, source.path)
        q.path.append(this)
        queue.append(q)
        visited[n.xCor][n.yCor] = True
    if this.yCor > 0 :
      n = arr[this.xCor][this.yCor - 1]
      if n.status == 1 and visited[n.xCor][n.yCor] == False:
        q = QItem(n, source.dist + 1, source.path)
        q.path.append(this)
        queue.append(q)
        visited[n.xCor][n.yCor] = True
    if this.yCor < D-1 :
      n = arr[this.xCor][this.yCor + 1]
      if n.status == 1 and visited[n.xCor][n.yCor] == False:
        q = QItem(n, source.dist + 1, source.path)
        q.path.append(this)
        queue.append(q)
        visited[n.xCor][n.yCor] = True
  return []

#Testing BFS
#arr = buildShipLayout()
#botLocation = placeBot(arr)
#leakLocation = placeLeak(arr, botLocation)
#printArray(arr, botLocation, leakLocation, None)
#path = shortestPath(arr, botLocation, leakLocation)
#print(f'path : {path}')

def sense(botLocation, leakLocation, secondLeak, leak1Found, leak2Found, k):
    # Boundaries for which the detection square can be, k is an inputted number
    x_min = max(botLocation.xCor - k, 0)
    x_max = min(botLocation.xCor + k + 1, D)
    y_min = max(botLocation.yCor - k, 0)
    y_max = min(botLocation.yCor + k + 1, D)

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if (i == leakLocation.xCor and j == leakLocation.yCor and leak1Found == False) :
                #Leak is in the Square of 2K
                return True
            if (not(secondLeak is None) and i == secondLeak.xCor and j == secondLeak.yCor and leak2Found == False):
                return True
    # No leak detected in teh Square of 2K
    return False
#Testing Sense
#arr = buildShipLayout()
#botLocation = placeBot(arr)
#leakLocation = placeLeak(arr, botLocation)
#printArray(arr, botLocation, leakLocation, None)
#sense(botLocation, leakLocation, None, 1)

def initializeProbability(ship, botLocation):
    count = 0
    for j in range(D):
      for i in range(D):
        if ship[i][j].status == 1:
            count += 1

    for j in range(D):
      for i in range(D):
        if ship[i][j].status == 1:
            ship[i][j].prob = 1 / count

def calculateprobability(ship, botLocation, k):
   for j in range(D):
     for i in range(D):
       x_min = max(i - k, 0)
       x_max = min(i + k + 1, D)
       y_min = max(j - k, 0)
       y_max = min(j + k + 1, D)
       if (ship[i][j].leakStatus == 0 or ship[i][j].status == 0 or ship[i][j].prob == 0):
        ship[i][j].prob = 0
       else:
        tot = 0
        leaked = 0
        for p in range(x_min,x_max):
         for q in range(y_min,y_max):
           tot += 1
           if (ship[p][q].
               status == 1 and ship[p][q].leakStatus == 1):
             leaked += 1
        ship[i][j].prob = (leaked/tot) / (D^2)
   return ship
#Testing pronbablity
#arr = buildShipLayout()
#botLocation = placeBot(arr)
#leakLocation = placeLeak(arr, botLocation)
#calculateprobability(arr, botLocation, 1)
#printArray(arr, botLocation, leakLocation, None)

def getHighestProb(ship):
    top = None
    for j in range(D):
      for i in range(D):
        if top == None or (ship[i][j].prob > top.prob and ship[i][j].leakStatus == 1 and ship[i][j].status == 1 ):
          top = ship[i][j]
    return top
#Testing highest
#arr = buildShipLayout()
#botLocation = placeBot(arr)
#leakLocation = placeLeak(arr, botLocation)
#secondLeak = placeSecondLeak(arr, botLocation, leakLocation)
#calculateprobability(arr, botLocation, 1)
#printArray(arr, botLocation, leakLocation, secondLeak)
#print(f'{getHighestProb(arr)}')

"""### Bot 1 & 2 Contained Below"""

def getNextOpen(ship, botLocation):
  printArray(ship,botLocation, leakLocation, None)
  newLocation = None
  dia = K
  found = False
  while (found == False):
    for j in range(botLocation.yCor-dia, botLocation.yCor+dia+1):
      for i in range(botLocation.xCor-dia, botLocation.xCor+dia+1):
        if ( i >= 0 and i < D and j >= 0 and j < D) and ship[i][j].status == 1 and ship[i][j].leakStatus == 1 :
          found = True
          return ship[i][j]
    dia += 1
    #print(f'dia {dia}')
  return None

def bot1(ship, botLocation, leakLocation, leak1Found, leak2Found, actions):
    senseLeak = False
    senseLeak = sense(botLocation, leakLocation, None, leak1Found, True, K)
    actions += 1;
    x_min = max(botLocation.xCor - K, 0)
    x_max = min(botLocation.xCor + K, D - 1)
    y_min = max(botLocation.yCor - K, 0)
    y_max = min(botLocation.yCor + K, D - 1)
    if senseLeak:
      for j in range(D):
        for i in range(D):
          if (i <x_min or i > x_max) or (j < y_min or j > y_max):
            ship[i][j].leakStatus = 0;
            #print(f'Sensed {i},{j} {ship[i][j]}')
    else :
      for j in range(y_min, y_max + 1):
        for i in range(x_min,x_max + 1):
          ship[i][j].leakStatus = 0;
          #print(f'No Sense {i},{j} {ship[i][j]}')
    botLocation.leakStatus = 0;
    botLocation.prob = 0
    calculateprobability(ship, botLocation, K)
    #print(f'leak sense is {senseLeak}')
    #printArray(ship,botLocation,leakLocation, None)
    return getNextOpen(ship, botLocation)

def bot2(ship, botLocation, leakLocation, leak1Found, leak2Found, actions):
    senseLeak = False
    senseLeak = sense(botLocation, leakLocation, None, leak1Found, True, K)
    actions += 1;
    x_min = max(botLocation.xCor - K, 0)
    x_max = min(botLocation.xCor + K, D - 1)
    y_min = max(botLocation.yCor - K, 0)
    y_max = min(botLocation.yCor + K, D - 1)
    if senseLeak:
      for j in range(D):
        for i in range(D):
          if (i <x_min or i > x_max) or (j < y_min or j > y_max):
            ship[i][j].leakStatus = 0;
            #print(f'Sensed {i},{j} {ship[i][j]}')
    else :
      for j in range(y_min, y_max + 1):
        for i in range(x_min,x_max + 1):
          ship[i][j].leakStatus = 0;
          #print(f'No Sense {i},{j} {ship[i][j]}')
    botLocation.leakStatus = 0;
    botLocation.prob = 0
    calculateprobability(ship, botLocation, K)
    #print(f'leak sense is {senseLeak}')
    #printArray(ship,botLocation,leakLocation)
    return getHighestProb(ship)

"""### Bot 3 & 4 Contained Below"""

def get_distance(arr, start, goal):

    if start == goal:
        return 0

    visited = [[False for i in range(D)] for j in range(D)]
    queue = []

    source = QItem(start, 0, [])
    queue.append(source)
    visited[start.xCor][start.yCor] = True

    while len(queue) != 0:
        source = queue.pop(0)
        this = source.node

        if this == goal:
            return source.dist  # Return the distance instead of the path

        # Check and enqueue adjacent cells
        if this.xCor > 0:
            n = arr[this.xCor - 1][this.yCor]
            if n.status == 1 and not visited[n.xCor][n.yCor]:
                queue.append(QItem(n, source.dist + 1, source.path))
                visited[n.xCor][n.yCor] = True

        if this.xCor < D-1:
            n = arr[this.xCor + 1][this.yCor]
            if n.status == 1 and not visited[n.xCor][n.yCor]:
                queue.append(QItem(n, source.dist + 1, source.path))
                visited[n.xCor][n.yCor] = True

        if this.yCor > 0:
            n = arr[this.xCor][this.yCor - 1]
            if n.status == 1 and not visited[n.xCor][n.yCor]:
                queue.append(QItem(n, source.dist + 1, source.path))
                visited[n.xCor][n.yCor] = True

        if this.yCor < D-1:
            n = arr[this.xCor][this.yCor + 1]
            if n.status == 1 and not visited[n.xCor][n.yCor]:
                queue.append(QItem(n, source.dist + 1, source.path))
                visited[n.xCor][n.yCor] = True

    return -1  # Return -1 if no path is found

def beep(ship, botLocation, leakLocation, alpha):

    distance = get_distance(ship, botLocation, leakLocation)

    #print(f"DISTANCE INSIDE BEEP{beep}")

    if distance == np.inf:
        print("Problem w/ distance")
        exit()

    if distance == 1:
        probability = 1
    else:
        probability = np.exp(-alpha * (distance - 1))

    random = np.random.random()

    #print(f"PROBABILITY OF BEEP{probability}")

    return random <= probability

def beep_probability_update(ship, botLocation, alpha):

    adjacent_cells_updated = False

    for j in range(D):
        for i in range(D):
            # Skip cells that are closed or already known not to contain a leak
            if ship[i][j].leakStatus == 0 or ship[i][j].status == 0 or ship[i][j].prob == 0:
                continue

            # Calculate the distance from the current cell to the bot's location
            distance = get_distance(ship, botLocation, ship[i][j])

            # Calculate the increase in probability based on distance
            if distance == 1:
                probability_increase = 1  # Maximum increase in probability
            else:
                probability_increase = np.exp(-alpha * (distance - 1))

            # Update the probability
            # Here, we scale up the probability by a factor related to the decrease in distance
            ship[i][j].prob *= probability_increase
            # Ensure the probability does not exceed 1
            ship[i][j].prob = min(ship[i][j].prob, 1.0)

    return ship

# Why is this giving me a syntax error? It runs?

def no_beep_probability_update(ship, botLocation, alpha):

    for j in range(D):
        for i in range(D):
            # Skip cells that are closed or already known not to contain a leak
            if ship[i][j].leakStatus == 0 or ship[i][j].status == 0 or ship[i][j].prob == 0:
                continue

            # Calculate the distance from the current cell to the bot's location
            distance = get_distance(ship, botLocation, ship[i][j])

            # If distance is 1 (adjacent), then probably no leak there
            if distance == 1:
                probability_decrease = 1  # Maximum decrease in probabilit
            else:
                # Decrease probability based on distance
                probability_decrease = np.exp(-alpha * (distance - 1))

            # Update the probability
            # The new probability is the old probability adjusted by the decrease factor
            ship[i][j].prob *= (1 - probability_decrease)

    return ship

def bot3(ship, botLocation, leakLocation, leak1Found, leak2Found, actions, alpha, lastVisited):

    # Check for a beep (probabilistic sensing)
    beepCheck = beep(ship, botLocation, leakLocation, alpha)
    actions += 1

    # Update probabilities based on whether a beep was heard or not
    if beepCheck:
        beep_probability_update(ship, botLocation, alpha)
    else:
        no_beep_probability_update(ship, botLocation, alpha)

    return getHighestProb(ship)

def should_move(ship, current_location):

    # Example condition: move if the highest probability cell is above a certain threshold
    highest_prob_cell = getHighestProb(ship)
    if highest_prob_cell.prob > 0.01:
        return True
    else:
        return False

def bot4(ship, botLocation, leakLocation, leak1Found, leak2Found, actions, alpha):

    count = 0

    while True:
        # Sense for a beep to update probabilities
        got_beep = beep(ship, botLocation, leakLocation, alpha)
        actions += 1

        # Update probabilities based on the beep
        if got_beep:
            beep_probability_update(ship, botLocation, alpha)
        else:
            no_beep_probability_update(ship, botLocation, alpha)

        # Check if the bot should move or stay in place
        if should_move(ship, botLocation):
            # Move to the cell with the highest probability of containing the leak
            return getHighestProb(ship)
        else:
            # Optionally stay in the same place to gather more data
            #print("Staying in place to gather more data")
            count += 1
            if count >= 3:
                return getHighestProb(ship)

# Testing Cell

ship = buildShipLayout()
botLocation = placeBot(ship)
leakLocation = placeLeak(ship, botLocation)

print(beep(ship, botLocation, leakLocation, alpha))

"""### Bots 5 & 6 Contained Below

"""

def bot5(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions):
    senseLeak = False
    senseLeak = sense(botLocation, leakLocation, secondLeak, leak1Found, leak2Found, K)
    actions += 1;
    x_min = max(botLocation.xCor - K, 0)
    x_max = min(botLocation.xCor + K, D - 1)
    y_min = max(botLocation.yCor - K, 0)
    y_max = min(botLocation.yCor + K, D - 1)
    if senseLeak:
      if (leak1Found or leak2Found):
        for j in range(D):
          for i in range(D):
            if (i <x_min or i > x_max) or (j < y_min or j > y_max):
              ship[i][j].leakStatus = 0;
              #print(f'Sensed {i},{j} {ship[i][j]}')
    else :
      for j in range(y_min, y_max + 1):
        for i in range(x_min,x_max + 1):
          ship[i][j].leakStatus = 0;
          #print(f'No Sense {i},{j} {ship[i][j]}')
    botLocation.leakStatus = 0;
    botLocation.prob = 0
    calculateprobability(ship, botLocation, K)
    #print(f'leak sense is {senseLeak}')
    #printArray(ship,botLocation,leakLocation, None)
    return getNextOpen(ship, botLocation)

def bot6(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions):
    senseLeak = False
    senseLeak = sense(botLocation, leakLocation, secondLeak, leak1Found, leak2Found, K)
    actions += 1;
    x_min = max(botLocation.xCor - K, 0)
    x_max = min(botLocation.xCor + K, D - 1)
    y_min = max(botLocation.yCor - K, 0)
    y_max = min(botLocation.yCor + K, D - 1)
    if senseLeak:
      if (leak1Found or leak2Found):
        for j in range(D):
          for i in range(D):
            if (i <x_min or i > x_max) or (j < y_min or j > y_max):
              ship[i][j].leakStatus = 0;
              #print(f'Sensed {i},{j} {ship[i][j]}')
    else :
      for j in range(y_min, y_max + 1):
        for i in range(x_min,x_max + 1):
          ship[i][j].leakStatus = 0;
          #print(f'No Sense {i},{j} {ship[i][j]}')
    botLocation.leakStatus = 0;
    botLocation.prob = 0
    calculateprobability(ship, botLocation, K)
    #print(f'leak sense is {senseLeak}')
    #printArray(ship,botLocation,leakLocation, None)
    return getHighestProb(ship)

"""### Bots 7, 8, & 9 Contained Below"""

def beep_two_leaks(ship, botLocation, firstLeak, secondLeak, leak1Found, leak2Found, alpha):
    if not leak1Found and not leak2Found:
        # Both leaks are unfound; choose the closer one
        distance_to_first_leak = get_distance(ship, botLocation, firstLeak)
        distance_to_second_leak = get_distance(ship, botLocation, secondLeak)
        distance = min(distance_to_first_leak, distance_to_second_leak)
    elif not leak1Found:
        # Only the first leak is unfound
        distance = get_distance(ship, botLocation, firstLeak)
    elif not leak2Found:
        # Only the second leak is unfound
        distance = get_distance(ship, botLocation, secondLeak)

    if distance == np.inf:
        print("Problem with distance calculation")
        exit()

    if distance == 1:
        probability = 1
    else:
        probability = np.exp(-alpha * (distance - 1))

    random_chance = np.random.random()

    return random_chance <= probability

def bot7(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, alpha):

    # Check for a beep (probabilistic sensing)
    beepCheck = beep_two_leaks(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, alpha)
    actions += 1

    # Update probabilities based on whether a beep was heard or not
    if beepCheck:
        beep_probability_update(ship, botLocation, alpha)
    else:
        no_beep_probability_update(ship, botLocation, alpha)

    return getHighestProb(ship)

def joint_initialize_probability_matrix(ship):
    D = len(ship)
    prob_matrix = {}
    open_cells = [(i, j) for i in range(D) for j in range(D) if ship[i][j].status == 1]

    for i in range(len(open_cells)):
        for j in range(i + 1, len(open_cells)):
            cell1 = open_cells[i]
            cell2 = open_cells[j]
            prob_matrix[(cell1[0], cell1[1], cell2[0], cell2[1])] = 1 / (len(open_cells) * (len(open_cells) - 1) / 2)

    return prob_matrix

def joint_update_probability_matrix(ship, prob_matrix, botLocation, alpha, beep):
    D = len(ship)
    for key in prob_matrix.keys():
        r1, c1, r2, c2 = key
        distance1 = get_distance(ship, botLocation, ship[r1][c1])
        distance2 = get_distance(ship, botLocation, ship[r2][c2])

        prob_beep1 = np.exp(-alpha * (distance1 - 1))
        prob_beep2 = np.exp(-alpha * (distance2 - 1))

        if beep:
            # Update probability based on hearing a beep
            prob_matrix[key] *= 1 - (1 - prob_beep1) * (1 - prob_beep2)
        else:
            # Update probability based on not hearing a beep
            prob_matrix[key] *= (1 - prob_beep1) * (1 - prob_beep2)

def decide_next_move(ship, prob_matrix, botLocation):
    D = len(ship)
    max_prob = 0
    target_cell = None

    # Find the cell pair with the highest probability of containing the leaks
    for (r1, c1, r2, c2), prob in prob_matrix.items():
        if prob > max_prob:
            max_prob = prob
            # Choose one of the cells in the pair for simplicity
            target_cell = ship[r1][c1]
            #print(f"Target cell is{target_cell}")

    # Plan the path to the target cell
    return target_cell

def update_prob_matrix_for_visited_cell(prob_matrix, visitedCell):

    if not isinstance(visitedCell, tuple):
        visitedCell = (visitedCell.xCor, visitedCell.yCor)

    for key in prob_matrix.keys():
        (r1, c1, r2, c2) = key
        #print(r1, c1)
        #print(f"Visited cell{visitedCell}")
        if (r1, c1) == visitedCell or (r2, c2) == visitedCell:
            prob_matrix[key] = 0  # Set probability to zero for pairs including the visited cell
            #print("yes yes")

    # Normalize the probabilities after the update
    normalize_probabilities(prob_matrix)

def normalize_probabilities(prob_matrix):
    total_prob = sum(prob_matrix.values())
    if total_prob > 0:
        for key in prob_matrix.keys():
            prob_matrix[key] /= total_prob

def bot8(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, prob_matrix, alpha):

    # Perform beep check and update probabilities
    beepCheck = beep_two_leaks(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, alpha)
    joint_update_probability_matrix(ship, prob_matrix, botLocation, alpha, beepCheck)

    return decide_next_move(ship, prob_matrix, botLocation)

def get_highest_probability(prob_matrix):
    return max(prob_matrix.values(), default=0)

def bot9(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, prob_matrix, alpha):
    max_beep_checks = 3
    beep_checks_done = 0
    probability_threshold = 0.5  # Set your desired threshold
    highest_probability = 0

    while beep_checks_done < max_beep_checks and highest_probability < probability_threshold:
        # Perform beep check and update probabilities
        beepCheck = beep_two_leaks(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, alpha)
        joint_update_probability_matrix(ship, prob_matrix, botLocation, alpha, beepCheck)
        highest_probability = get_highest_probability(prob_matrix)

        beep_checks_done += 1

    return decide_next_move(ship, prob_matrix, botLocation)

"""### Main Simulation Loop"""

def execute_bot_strategy(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, strategy, actions, prob_matrix):
    if strategy == 1:
        return bot1(ship, botLocation, leakLocation, leak1Found, leak2Found, actions)
    elif strategy == 2:
        return bot2(ship, botLocation, leakLocation, leak1Found, leak2Found, actions)
    elif strategy == 3:
        return bot3(ship, botLocation, leakLocation, leak1Found, leak2Found, actions, alpha)
    elif strategy == 4:
        return bot4(ship, botLocation, leakLocation, leak1Found, leak2Found, actions, alpha)
    elif strategy == 5:
        return bot5(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions)
    elif strategy == 6:
        return bot6(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions)
    elif strategy == 7:
        return bot7(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, alpha)
    elif strategy == 8:
        return bot8(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, prob_matrix, alpha)
    elif strategy == 9:
        return bot9(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, actions, prob_matrix, alpha)

actions = 0

def run_simulation(alpha, strategy_number):
    count = 0
    global botLocation
    global leakLocation
    global actions
    ship = buildShipLayout()
    botLocation = placeBot(ship)
    print(botLocation)
    leakLocation = placeLeak(ship, botLocation)
    print(leakLocation)
    secondLeak = placeSecondLeak(ship, botLocation, leakLocation)
    time_step = 0
    time_out = 5000
    actions = 0
    initializeProbability(ship, botLocation)
    prob_matrix = joint_initialize_probability_matrix(ship)
    #printArray(ship,botLocation,leakLocation, secondLeak)
    leak1Found = False
    leak2Found = False
    lastVisited = (botLocation.xCor, botLocation.yCor)  # Initialize lastVisited
    while time_step < time_out:
        if botLocation == leakLocation and leak1Found == False:
            print(f'Leak 1 Location is {leakLocation}')
            leak1Found = True
            leakLocation.prob = 0
        if ((botLocation == secondLeak and leak2Found == False) or strategy_number < 5):
            if (strategy_number >= 5):
              print(f'Leak 2 Location is {secondLeak}')
              leakLocation.prob = 0
            leak2Found = True
        if (leak1Found and leak2Found):
            print("Bot has reached the leaks; Success")
            print(f"Actions{actions}")
            return "success", actions
        #Bot Move; Can be copy pasted or rewritten doesn't matter too much
        newLocation = execute_bot_strategy(ship, botLocation, leakLocation, secondLeak, leak1Found, leak2Found, strategy_number, actions, prob_matrix)
        if newLocation == lastVisited:
            count += 1
            if count == 5:
                print(f"Actions{actions}")
                return "success", actions
        #print(f'moving bot to {newLocation}')
        path = shortestPath(ship,botLocation, newLocation)
        #print(f'path : {path}')
        for i in range(len(path)):
          botLocation = path[i]
          actions += 1
          if (botLocation == leakLocation and leak1Found == False):
            leak1Found = True
            botLocation.leakStatus = 0
            botLocation.prob = 0
            if strategy_number >= 8:
                update_prob_matrix_for_visited_cell(prob_matrix, (botLocation.xCor, botLocation.yCor))
            print(f'Leak 1 Location is {leakLocation}')
          if ((botLocation == secondLeak and leak2Found == False) or strategy_number < 5):
            leak2Found = True
            botLocation.leakStatus == 0
            botLocation.prob = 0
            if strategy_number >= 8:
                update_prob_matrix_for_visited_cell(prob_matrix, (botLocation.xCor, botLocation.yCor))
            if (strategy_number >= 5):
              print(f'Leak 2 Location is {secondLeak}')
          if (leak1Found and leak2Found):
            print("Bot has plugged the leaks; Success")
            print(f"Actions{actions}")
            return "success", actions
          else:
            botLocation.leakStatus = 0
            botLocation.prob = 0
            if strategy_number >= 8:
                update_prob_matrix_for_visited_cell(prob_matrix, (botLocation.xCor, botLocation.yCor))
        botLocation = newLocation

        #printArray(ship,botLocation, leakLocation, secondLeak)
        time_step += 1
        #print(f'actions = {actions}')
    return "success", actions

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
alpha_values = [0.00416667, 0.00833333, 0.0125, 0.01666667, 0.02083333, 0.025, 0.02916667, 0.03333333, 0.0375, 0.04166667, 0.04583333, 0.05, 0.05416667, 0.05833333, 0.0625, 0.06666667, 0.07083333, 0.075, 0.07916667, 0.08333333, 0.0875, 0.09166667, 0.09583333, 0.1]
num_trials = 3

strategy_number = 1  # Strategy for bot

average_actions_per_K = {}
average_actions_per_alpha = {}

# # for K in k_values:
# #     total_actions = 0
# #     for _ in range(num_trials):
# #         _, actions_taken = run_simulation(K, strategy_number)
# #         total_actions += actions_taken
# #     average_actions = total_actions / num_trials
# #     average_actions_per_K[K] = average_actions  # Store average for each K

# Loop for alpha_values
for alpha in alpha_values:
    total_actions = 0
    for _ in range(num_trials):
        _, actions_taken = run_simulation(alpha, strategy_number)
        total_actions += actions_taken
    average_actions = total_actions / num_trials
    average_actions_per_alpha[alpha] = average_actions  # Store average for each alpha


# for K, avg_actions in average_actions_per_K.items():
#     print(f"K = {K}: Average Actions = {avg_actions}")

for alpha, avg_actions in average_actions_per_alpha.items():
    print(f"Alpha = {alpha}: Average Actions = {avg_actions}")
