from z3 import *

def solve_cloud_resource_allocation(D, V, L, C, k, alpha=1.0, beta=1.0, gamma=None):
    """
    Solve the cloud resource allocation optimization problem using Z3.

    Parameters:
    - D: List of predicted demand (requests/sec) for each region j
    - V: List of predicted variance of demand for each region j
    - L: Matrix of network latency from region i to users in region j (L[i][j])
    - C: List of cost per unit resource allocation in region i
    - k: Capacity (requests/sec) provided per unit of allocated resource
    - alpha: Weight for operational cost in objective function
    - beta: Weight for latency in objective function
    - gamma: Risk-aversion parameter for variance buffer, if none then optimised

    Returns:
    - model: Z3 model with solution
    - A: Resource allocation for each region
    - x: Demand serving fractions (x[i][j] = fraction of demand from region j served by region i)
    """

    N = len(D)
    assert len(V) == N, "Variance list V must have same length as demand list D"
    assert len(C) == N, "Cost list C must have same length as demand list D"
    assert len(L) == N and all(len(row) == N for row in L), "Latency matrix L must be N x N"

    print(f"Setting up optimization for {N} regions")
    print(f"Demands: {D}")
    print(f"Variances: {V}")
    print(f"Costs: {C}")
    print(f"Capacity per unit: {k}")

    opt = Optimize()

    if gamma is None:
        gamma = Real('gamma')
        opt.add(gamma <= 1, gamma >= 0)
    else:
        gamma = float(min(max(gamma, 0), 1))
        print(f"Risk aversion parameter: {gamma}")

    # Decision Variables
    # A_i: Amount of resource to allocate in region i
    A = [Real(f'A_{i}') for i in range(N)]

    # x_ij: Fraction of demand from region j served by resources in region i
    x = [[Real(f'x_{i}_{j}') for j in range(N)] for i in range(N)]

    # Objective is to minimize weighted sum of cost and latency
    # Cost component:= alpha * sum (C_i * A_i)
    cost_component = Sum([C[i] * A[i] for i in range(N)])

    # Latency component:= beta * (sum over i,j {x_ij * D_j * L_ij})
    latency_component = Sum([x[i][j] * D[j] * L[i][j] for j in range(N) for i in range(N)])

    objective = alpha * cost_component + beta * latency_component

    # Capacity Constraint:= sum (k * A_i) >= sum(D_j + gamma * V_j)
    total_capacity = Sum([k * A[i] for i in range(N)])
    total_demand_with_buffer = Sum([D[j] + gamma * V[j] for j in range(N)])
    opt.add(total_capacity >= total_demand_with_buffer)

    # Demand Satisfaction:= sum xij = 1, for all j
    for j in range(N):
        opt.add(Sum([x[i][j] for i in range(N)]) == 1)

    # Regional Capacity Limit: sum (xij * D_j) <= k * A_i for all "i"
    for i in range(N):
        opt.add(Sum([x[i][j] * D[j] for j in range(N)]) <= k * A[i])

    for i in range(N):
        opt.add(A[i] >= 0)
        for j in range(N):
            opt.add(x[i][j] >= 0)
            opt.add(x[i][j] <= 1)  # Fractions cannot exceed 1

    opt.minimize(objective)

    print("\nSolving optimization problem...")
    result = opt.check()

    if result == sat:
        model = opt.model()
        print("Optimal solution found!")

        A_values = [float(model[A[i]].as_fraction()) if model[A[i]] is not None else 0 for i in range(N)]
        x_values = [[float(model[x[i][j]].as_fraction()) if model[x[i][j]] is not None else 0
                    for j in range(N)] for i in range(N)]

        if not isinstance(gamma, float):
            gamma = float(model[gamma].as_fraction())

        total_cost = sum(C[i] * A_values[i] for i in range(N))
        total_latency = sum(x_values[i][j] * D[j] * L[i][j] for j in range(N) for i in range(N))
        objective_value = alpha * total_cost + beta * total_latency

        print(f"\nOptimal objective value: {objective_value:.4f}")
        print(f"Total operational cost: {total_cost:.4f}")
        print(f"Total weighted latency: {total_latency:.4f}")
        print(f"Risk-aversion: {gamma}")
        print(f"\nResource allocations (A):")
        for i in range(N):
            print(f"  Region {i}: {A_values[i]:.4f} units")

        print(f"\nDemand serving fractions (x):")
        print("    " + " | ".join([f"Reg{j:2}" for j in range(N)]))
        for i in range(N):
            row_str = f"R{i}: "
            for j in range(N):
                row_str += f"{x_values[i][j]:5.2f} |"
            print(row_str)

        print(f"\nConstraint verification:")
        total_cap = sum(k * A_values[i] for i in range(N))
        total_req = sum(D[j] + gamma * V[j] for j in range(N))
        print(f"Total capacity: {total_cap:.4f}, Required capacity: {total_req:.4f}")
        print(f"Capacity constraint satisfied: {total_cap >= total_req}")

        for j in range(N):
            demand_sum = sum(x_values[i][j] for i in range(N))
            print(f"Demand satisfaction region {j}: {demand_sum:.4f} (should be 1.0)")

        return model, A_values, x_values, objective_value
    else:
        print("No solution found")
        return None, None, None, None


if __name__ == "__main__":
    # Test the implementation with sample data
    print("Testing the cloud resource allocation optimizer with sample data:")
    print("=" * 60)

    # Sample data for 3 regions
    N = 3
    D_sample = [100, 150, 80]  # Demand in requests/sec
    V_sample = [10, 20, 8]     # Variance
    C_sample = [1.0, 1.2, 0.8] # Cost per unit resource
    k_sample = 50              # Capacity per unit resource
    L_sample = [               # Latency matrix (region i to users in region j)
        [10, 50, 80],          # From region 0
        [50, 10, 30],          # From region 1
        [80, 30, 10]           # From region 2
    ]
    
    model, A_opt, x_opt, obj_value = solve_cloud_resource_allocation(
        D_sample, V_sample, L_sample, C_sample, k_sample,
        alpha=1.0, beta=0.1, gamma=0.115
    )