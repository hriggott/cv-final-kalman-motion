import numpy as np
import matplotlib.pyplot as plt

#Method that calculates l2 norm for difference of two vectors
def getNorm(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

#Method that calculates l2 norm for difference of two vectors over time
def getNormOverTime(x_actual,y_actual,x_estimated,y_estimated):
    vals = []
    for i in range (len(x_actual)):
        vals.append(getNorm([x_actual[i],y_actual[i]],[x_estimated[i],y_estimated[i]]))
    return vals

#Method that returns the error integrated over time of some x and y vs x actual and y actual. x and y can be a list of positions that are measured, corrected, or predicted.
def getIntegratedError(x_actual,y_actual,x,y,t):
    timestep = t[1]-t[0]
    
    x_error_integrated = 0
    y_error_integrated = 0
    for i in range (0,(len(t)-1)):
        x_error_integrated += np.abs((x[i]+x[i+1]-x_actual[i]-x_actual[i+1])*timestep/2)
        y_error_integrated += np.abs((y[i]+y[i+1]-y_actual[i]-y_actual[i+1])*timestep/2)
    return x_error_integrated,y_error_integrated

#Method that creates a subplot containing positions vs time, errors vs time, and y positions vs x positions. x_estimate and y_estimate can be either predicted or corrected.
def getPlots(x_measured,y_measured,x_actual,y_actual,x_estimate,y_estimate,t,p_or_c):
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413)
    ax4 = plt.subplot(414)
    fig.tight_layout()
    
    #First plot is of the measured position, actual position, and estimated or predicted position for x and y vs time.
    line, = ax1.plot(t,x_measured,label='Measurement Signal x')
    ax1.legend()
    line, = ax1.plot(t,y_measured,label='Measurement Signal y')
    ax1.legend()
    line, = ax1.plot(t,x_actual,label='Actual trajectory x')
    ax1.legend()
    line, = ax1.plot(t,y_actual,label='Actual trajectory y')
    ax1.legend()
    line, = ax1.plot(t,x_estimate,label=p_or_c.capitalize()+' trajectory x')
    ax1.legend()
    line, = ax1.plot(t,y_estimate,label=p_or_c.capitalize()+' trajectory y')
    ax1.legend()
    ax1.set_title('Positions vs Time')
    
    #Second plot is of the error between the actual position and the measurement and the error between the estimated position and
    #measurement, for both x and y.
    line, = ax2.plot(t, np.abs(x_actual-x_measured),label='Error between actual trajectory and measured trajectory x')
    ax2.legend()
    line, = ax2.plot(t, np.abs(y_actual-y_measured),label='Error between actual trajectory and measured trajectory y')
    ax2.legend()
    line, = ax2.plot(t,np.abs(x_actual-x_estimate),label='Error between actual trajectory and '+p_or_c+' measurement x')
    ax2.legend()
    line, = ax2.plot(t,np.abs(y_actual-y_estimate),label='Error between actual trajectory and '+p_or_c+' measurement y')
    ax2.legend()
    ax2.set_title('Errors vs time')
    
    #Third plot is of y position vs x position for the measured, estimated, and actual positions.This is the visual representation
    #of the motion of the partical.
    line, = ax3.plot(x_measured,y_measured,label='Measured Position')
    ax3.legend()
    line, = ax3.plot(x_estimate,y_estimate,label=p_or_c.capitalize()+' Position')
    ax3.legend()
    line, = ax3.plot(x_actual,y_actual,label='Actual position')
    ax3.legend()
    ax3.set_title('x positions vs y positions')
             
    #Fourth plot is l2 norm of the difference between estimated position and actual position
    line, = ax4.plot(t,getNormOverTime(x_actual,y_actual,x_estimate,y_estimate))
    ax4.set_title('L2 Norm of Error between Actual Position and '+p_or_c.capitalize()+' Position over time')
    
#Generates bar graph of integrated errors, and subplots for predicted and corrected positions. The accuracy of the predicted positions will determine how good the object detection tracking is, while the accuracy of the corrected positions are a measure of how well the Kalman filter is performing.
def analyzeResults(x_actual,y_actual,x_measured,y_measured,x_predicted,y_predicted,x_corrected,y_corrected,t):
    timestep = t[1]-t[0]
    
    x_error_measured,y_error_measured = getIntegratedError(x_actual,y_actual,x_measured,y_measured,t)
    x_error_predicted,y_error_predicted = getIntegratedError(x_actual,y_actual,x_predicted,y_predicted,t)
    x_error_corrected,y_error_corrected = getIntegratedError(x_actual,y_actual,x_corrected,y_corrected,t)
    
    legend = ['x error measured','y error measured','x error predicted','y error predicted','x error corrected','y error corrected']
    data = [x_error_measured,y_error_measured,x_error_predicted,y_error_predicted,x_error_corrected,y_error_corrected]
    fig = plt.figure(figsize=(12,6))
    fig.tight_layout()
    plt.bar(legend,data)
    plt.title('Integral of x and y errors over time')
    plt.show()
    #Plots for predicted positions
    getPlots(x_measured,y_measured,x_actual,y_actual,x_predicted,y_predicted,t,'predicted')
    #Plots for corrected positions
    getPlots(x_measured,y_measured,x_actual,y_actual,x_corrected,y_corrected,t,'corrected')
    
