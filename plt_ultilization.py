

keep_alive_time_ultilization =[0.8903081662087513,0.427668335566078,0.9999063981631117]
keep_alive_memory_ultilization = [0.7856720131990306,0.4276683355660779,0.0019006705631251385]

ARIMA_time_ultilization = [0.8592320710731093,0.6075332263489002,1.0]
ARIMA_memory_ultilization = [0.8027581389060878,0.006022863557380346,0.0019143565323917117]

LSTM_time_ultilization = [0.8629219458979132,0.5884394983532288,0.9038310089297148]
LSTM_memory_ultilization = [0.7997009332189975,0.37106256461473325,0.269434685876522]


# 三种方案的时间利用率折线图、资源利用率折线图

import matplotlib.pyplot as plt

# # Plotting
# plt.plot(keep_alive_time_ultilization, label='Keep Alive Time Utilization')
# plt.plot(ARIMA_time_ultilization, label='ARIMA Time Utilization')
# plt.plot(LSTM_time_ultilization, label='LSTM Time Utilization')
# plt.xlabel('Time')
# plt.ylabel('Utilization')
# plt.title('Time Utilization Comparison')
# plt.legend()
# plt.show()

# # Time Utilization Comparison
# plt.subplot(1, 2, 1)
# plt.plot(keep_alive_time_ultilization, 'r-', marker='o', label='Keep Alive')
# plt.plot(ARIMA_time_ultilization, 'g-', marker='o', label='ARIMA')
# plt.plot(LSTM_time_ultilization, 'b-', marker='o', label='LSTM')
# plt.xlabel('arrival time interval',fontsize=15)
# plt.xticks([])
# plt.ylabel('Utilization',fontsize=15)
# plt.title('Time Utilization Comparison')
# plt.legend()

# # Memory Utilization Comparison
# plt.subplot(1, 2, 2)
# plt.plot(keep_alive_memory_ultilization, 'r-', marker='s', label='Keep Alive')
# plt.plot(ARIMA_memory_ultilization, 'g-', marker='s', label='ARIMA')
# plt.plot(LSTM_memory_ultilization, 'b-', marker='s', label='LSTM')
# plt.xlabel('arrival time interval',fontsize=15)
# plt.xticks([])
# plt.ylabel('Utilization',fontsize=15)
# plt.title('Memory Utilization Comparison')
# plt.legend()
# plt.tight_layout()

# plt.savefig('ultilization.png')


plt.plot(keep_alive_time_ultilization, 'r-', marker='o', label='Keep Alive time ultilization')
plt.plot(keep_alive_memory_ultilization, 'r-', marker='s', label='Keep Alive memory ultilization')
plt.plot(ARIMA_time_ultilization, 'g-', marker='o', label='ARIMA time ultilization')
plt.plot(ARIMA_memory_ultilization, 'g-', marker='s', label='ARIMA memory ultilization')
plt.plot(LSTM_time_ultilization, 'b-', marker='o', label='LSTM time ultilization')
plt.plot(LSTM_memory_ultilization, 'b-', marker='s', label='LSTM memory ultilization')

plt.xlabel('arrival time interval',fontsize=15)
plt.xticks([])
plt.ylabel('Utilization',fontsize=15)
plt.legend()

plt.title('Utilization Comparison')
plt.legend()
# plt.tight_layout()

plt.savefig('ultilization_Comparison.png')

