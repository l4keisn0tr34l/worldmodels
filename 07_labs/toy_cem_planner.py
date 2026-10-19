"""CEM MPC for the learned 1D world model. Uses no external packages.
Run: python3 07_labs/toy_cem_planner.py
"""
import math
import random
from toy_world_model import LinearWorldModel, State, collect_data, true_step


def imagined_cost(model, s, seq, target):
    current = s
    effort = 0.0
    running_cost = 0.0
    for a in seq:
        current = model.predict(current, a)
        effort += a*a
        running_cost += (current.x - target)**2
    return (current.x-target)**2 + 0.35 * current.v**2 + 0.025 * running_cost + 0.008 * effort


def cem_plan(model, state, target, horizon=16, population=180, iterations=6, elite=20, seed=0):
    rng = random.Random(seed)
    mean = [0.0] * horizon
    std = [0.85] * horizon
    best_seq = None
    best_cost = float('inf')
    for _ in range(iterations):
        ranked = []
        for _ in range(population):
            seq = [max(-1.0, min(1.0, rng.gauss(m, max(0.05, sd)))) for m,sd in zip(mean,std)]
            cost = imagined_cost(model, state, seq, target)
            ranked.append((cost, seq))
            if cost < best_cost:
                best_cost, best_seq = cost, seq
        ranked.sort(key=lambda item: item[0])
        selected = [seq for _,seq in ranked[:elite]]
        for t in range(horizon):
            vals = [seq[t] for seq in selected]
            m = sum(vals)/elite
            v = sum((x-m)**2 for x in vals)/elite
            mean[t] = 0.25*mean[t] + 0.75*m
            std[t] = max(0.05, 0.25*std[t] + 0.75*math.sqrt(v))
    return best_seq, best_cost


def run_closed_loop(model, target=3.0, steps=22, seed=12):
    env_rng = random.Random(seed)
    state = State(0.0, 0.0)
    history = [(0, state.x, state.v, 0.0)]
    for t in range(1,steps+1):
        # Replan every real action. Unlike open-loop execution, it observes each new state.
        action_seq, _ = cem_plan(model, state, target, horizon=16, seed=seed+t)
        action = action_seq[0]
        state = true_step(state, action, env_rng)
        history.append((t, state.x, state.v, action))
    return state, history


def main():
    model = LinearWorldModel().fit(collect_data())
    state, history = run_closed_loop(model)
    print('Move a point mass from x=0 to x=3, starting at rest.')
    print('At each step choose a 16-step future but EXECUTE ONLY FIRST ACTION.')
    print('step | position | velocity | action')
    for t,x,v,a in history:
        if t % 2 == 0 or t==1:
            print(f'{t:4d} | {x:8.3f} | {v:8.3f} | {a:7.3f}')
    print(f'Final position: {state.x:.3f}, abs(goal-position): {abs(3.0-state.x):.3f}')
    print('Compare zero-action baseline: starting from rest remains near x=0 (apart from noise).')
    print('Try using a deliberately wrong dynamics model and observe planner exploitation.')


if __name__ == '__main__':
    main()
