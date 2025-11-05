import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def format_axis(ax):
    """
    Formats the axis.
    
    Parameters:

    - ax: axis to be formated.
    
    Returns:
    None.
    """
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.12f}'))
    
def plot_fidelity(a_values,b_values,alpha_values,fid_T,fid_R,flag = 'normal'):

    """
    Plot fidelity as a function of the given parameters.
    
    Parameters:

    - a_values: The value or the array of the values of a.
    - b_values: The value or the array of the values of b (if provided with an empty array, the function uses b = sqrt(1 - |a|^2)).
    - alpha_values: An array of the values of alpha.
    - fid_T: Fidelity of the transmitted arm.
    - fid_R: Fidelity of the reflected arm.
    - flag: Determines the X axis of the plot. flag : {'normal', 'square'}. Determines the X-axis of the plot. "normal": uses (a, alpha) as axes. "square": uses (|a|², μ) as axes.
    
    Returns:
    None. The function saves the fidelity plot under the name "plot.png" and displays it.
    """
    
    plt.rcParams['axes.formatter.useoffset'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = False

    if(isinstance(alpha_values, np.ndarray)):
        if(not isinstance(b_values, np.ndarray) or b_values.size == 0):
            if(flag == 'normal'):
                title_text = rf'$\alpha$ for $a$ = {a_values:.8f} and $b$ = {b_values:.8f}'
                xaxis = alpha_values
                xaxis_title = r'$\alpha$'
            elif(flag == 'square'):
                title_text = rf'$\mu$ for $a$ = {a_values:.8f} and $b$ = {b_values:.8f}'
                xaxis = np.abs(alpha_values) ** 2
                xaxis_title = r'$\mu$'
            else:
                print(f"Incorrect flag entered: " + flag + ". The allowed flags are normal and square.")
                exit(0)
        else:
            if(flag == 'normal'):
                title_text = r'$\alpha$ for $a$ = ' f'{a_values}'
                xaxis = alpha_values
                xaxis_title = r'$\alpha$'
            elif(flag == 'square'):
                title_text = r'$\mu$ for $a$ = ' f'{a_values}'
                xaxis = np.abs(alpha_values) ** 2
                xaxis_title = r'$\mu$'
            else:
                print(f"Incorrect flag entered: " + flag + ". The allowed flags are normal and square.")
                exit(0)

    elif(isinstance(a_values, np.ndarray)):
        if(flag == 'normal'):
            title_text = rf'$a$ for $\mu$ = {alpha_values**2}'
            xaxis = a_values
            xaxis_title = r'$a$'
        elif(flag == 'square'):
            title_text = rf'$|a|^2$ for $\mu$ = {alpha_values**2}'
            xaxis = np.abs(a_values) ** 2
            xaxis_title = r'$|a|^2$'
        else:
            print(f"Incorrect flag entered: " + flag + ". The allowed flags are normal and square.")
            exit(0)
    else:
        title_text = '???'
    
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(wspace=0.5)
    plt.subplot(1,2,1)
    plt.plot(xaxis, np.real(fid_T), linewidth=1.5)
    plt.title(r'$F_{T}$ vs ' + title_text)
    plt.xlabel(xaxis_title)
    plt.ylabel('Fidelity')
    plt.grid(True)
    ax = plt.gca()
    format_axis(ax)
    #ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))

    plt.subplot(1,2,2)
    plt.plot(xaxis, np.real(fid_R), linewidth=1.5)
    plt.title(r'$F_{R}$ vs ' + title_text)
    plt.xlabel(xaxis_title)
    plt.ylabel('Fidelity')
    plt.grid(True)
    ax = plt.gca()
    format_axis(ax)
    #ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))


    plt.savefig("plot.png")
    plt.show()