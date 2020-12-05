import numpy as np

#from LKBB import Linear_Kalman_Black_Box
class Linear_Kalman_Black_Box:
    #Create a new Kalman class. Constructor takes the time step, dt, the std dev of acceleration (Sigma_q), and the std dev of the mesurement (Sigma_z). 
    #Sigma_q and Sigma_z are parameters that need to be tuned, so you may have to play around with them for the filter to work well.

    def __init__(self,dt,Sigma_q,Sigma_z):
        self.dt = dt
        #The A matrix is the matrix of the dynamics, such that x_k+1 = A*x_k. Because our state vector is (x,x_dot,x_dot_dot,y,y_dot,y_dot_dot), the matrix 
        #Is made up of a 3x3 in the upper left corner, and the same 3x3 in the lower right corner. A_sub defines this 3x3 matrix.
        A_sub = np.array([[1,dt,.5*dt**2],[0,1,dt],[0,0,1]])
        self.A = np.zeros([6,6])
        self.A[0:3,0:3] = A_sub
        self.A[3:6,3:6] = A_sub
        #Covariance matrix
        self.P = np.identity(6);
        #Process noise covariance. Similar to A matrix, it is made up of the same 3x3 matrix in the upper left corner and lower right corner.
        Q_sub = np.array([[dt**4/4, dt**3/2, dt**2/2],[dt**3/2, dt**2, dt],[dt**2/2, dt, 1]]) * Sigma_q
        self.Q = np.zeros([6,6])
        self.Q[0:3,0:3] = Q_sub
        self.Q[3:6,3:6] = Q_sub
        #Measurement noise covariance. 
        self.R = np.identity(2)*Sigma_z;
        #Transformation Matrix. When multiplied by state vector, it extracts the x position and y position.
        self.H = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
        #List to store corrected x positions
        self.x_pos = []
        #List to store corrected y positions
        self.y_pos = []
        #Variable to store previous state
        self.prev_state = np.zeros([6,1])
        #Variable to store predicted state
        self.x_k_predict = np.zeros([6,1])

    #Method for debugging purposes
    def getParams(self):
        print("A")
        print(self.A)
        print("P")
        print(self.P)
        print("Q")
        print(self.Q)
        print("R")
        print(self.R)
        print("H")
        print(self.H)

    #Predict next state, error covariance, Kalman gain, and return the predicted x and y positions.
    def predict(self):
        self.x_k_predict = self.A@self.prev_state
        self.P = self.A@self.P@np.transpose(self.A)+self.Q
        self.KalmanGain = self.P@np.transpose(self.H)@np.linalg.inv(self.H@self.P@np.transpose(self.H)+self.R)
        return self.x_k_predict[0,0],self.x_k_predict[3,0]

    #Update prediction
    def update(self,x_pos_measured,y_pos_measured):
        self.prev_state[:,:] = self.x_k_predict+self.KalmanGain@(np.array([[x_pos_measured],[y_pos_measured]])-self.H@self.x_k_predict)
        self.P = (np.identity(len(self.KalmanGain@self.H))-self.KalmanGain@self.H)@self.P
        self.x_pos.append(self.prev_state[0,0])
        self.y_pos.append(self.prev_state[3,0])

    #Call this method when the first measurement is computed. It stores the first measurement as the first values for x position and y position, since
    #there isn't enough information to correct the measurement at this point.
    def setInitialPosition(self,x_pos_measured,y_pos_measured):
        self.prev_state[0,0] = x_pos_measured
        self.prev_state[3,0] = y_pos_measured
        self.x_pos.append(x_pos_measured)
        self.y_pos.append(y_pos_measured)

    #Returns the lists of corrected x and y positions. Used for analysis of algorithm.
    def getCorrectedPositions(self):
        return self.x_pos, self.y_pos