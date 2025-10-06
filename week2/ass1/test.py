
import irsim
#from custom_behavior_methods_2 import metrics_p

env = irsim.make("circle.yaml")
env.load_behavior("custom_behavior_methods_2")

for _ in range(5000):
    env.step()
    env.render(0.03)
    if env.done():
        break

#metrics_p()
env.end(3)







