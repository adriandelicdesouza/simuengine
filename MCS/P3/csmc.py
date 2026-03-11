import numpy as np

# ============================================================================
# !!!!! CONFIGURATION: ALL PARAMETERS HERE !!!!!
# ============================================================================
# EDIT THESE PARAMETERS TO CHANGE THE SIMULATIONS
# All code below uses these values - change once, affects everything
# ============================================================================

# Set seed for reproducibility (comment out for different results each run)
np.random.seed(42)

# --- COST & SALES PARAMETERS ---
SALES_SIMULATIONS = 10000         # Number of Monte Carlo runs
SALES_MIN_UNITS = 100             # Minimum sales (worst case)
SALES_MAX_UNITS = 500             # Maximum sales (best case)
SALES_AVG_UNITS = 300             # Expected average sales (most likely)
SALES_PRICE = 50                  # Selling price per unit ($)

COST_MIN = 15                      # Minimum product cost per unit ($)
COST_MAX = 25                      # Maximum product cost per unit ($)
COST_AVG = 20                      # Average product cost ($)

FIXED_COSTS = 5000                # Fixed overhead costs ($)

# ============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW
# ============================================================================


def simulate_cost_and_sales():
    """
    Run Monte Carlo simulation for cost and sales uncertainty
    
    Returns:
        Dictionary with all simulation results and metrics
    """
    
    # Generate sales volumes using triangular distribution
    # (worst case, most likely, best case)
    sales_volumes = np.random.triangular(
        left=SALES_MIN_UNITS,
        mode=SALES_AVG_UNITS,
        right=SALES_MAX_UNITS,
        size=SALES_SIMULATIONS
    )
    
    # Generate product costs using normal distribution
    # (centered around average with some variation)
    product_costs = np.random.normal(
        loc=COST_AVG,
        scale=2,  # Standard deviation of $2
        size=SALES_SIMULATIONS
    )
    
    # Ensure costs stay within realistic bounds
    product_costs = np.clip(product_costs, COST_MIN, COST_MAX)
    
    # Calculate financial metrics for each simulation
    revenues = sales_volumes * SALES_PRICE
    variable_costs = sales_volumes * product_costs
    total_costs = variable_costs + FIXED_COSTS
    profits = revenues - total_costs
    
    # Calculate statistics
    stats = {
        'mean_profit': float(profits.mean()),
        'median_profit': float(np.median(profits)),
        'std_profit': float(profits.std()),
        'min_profit': float(profits.min()),
        'max_profit': float(profits.max()),
        'p5_profit': float(np.percentile(profits, 5)),
        'p25_profit': float(np.percentile(profits, 25)),
        'p75_profit': float(np.percentile(profits, 75)),
        'p95_profit': float(np.percentile(profits, 95)),
        'mean_revenue': float(revenues.mean()),
        'mean_variable_costs': float((revenues - profits - FIXED_COSTS).mean()),
        'mean_total_costs': float(total_costs.mean()),
        'mean_sales_volume': float(sales_volumes.mean()),
        'mean_product_cost': float(product_costs.mean()),
    }
    
    # Calculate risk metrics
    prob_loss = (profits < 0).sum() / SALES_SIMULATIONS
    prob_profit_5k = (profits > 5000).sum() / SALES_SIMULATIONS
    prob_profit_10k = (profits > 10000).sum() / SALES_SIMULATIONS
    
    risk_metrics = {
        'prob_loss': float(prob_loss),
        'prob_profit_5k': float(prob_profit_5k),
        'prob_profit_10k': float(prob_profit_10k),
    }
    
    return {
        'sales_volumes': sales_volumes,
        'product_costs': product_costs,
        'revenues': revenues,
        'total_costs': total_costs,
        'profits': profits,
        'stats': stats,
        'risk_metrics': risk_metrics
    }


def print_detailed_results(results):
    """Print comprehensive analysis of simulation results"""
    
    stats = results['stats']
    risk = results['risk_metrics']
    
    # Header
    print("\n" + "="*70)
    print("COST & SALES MONTE CARLO SIMULATION - COMPLETE ANALYSIS")
    print("="*70)
    
    # Input Summary
    print("\n" + "-"*70)
    print("INPUT PARAMETERS")
    print("-"*70)
    print(f"Number of Simulations:      {SALES_SIMULATIONS:>10,}")
    print(f"Sales Volume Range:         {SALES_MIN_UNITS:>10} - {SALES_MAX_UNITS} units")
    print(f"Most Likely Sales:          {SALES_AVG_UNITS:>10} units")
    print(f"Product Cost Range:         ${COST_MIN:>9} - ${COST_MAX}/unit")
    print(f"Average Product Cost:       ${COST_AVG:>9}/unit")
    print(f"Selling Price:              ${SALES_PRICE:>9}/unit")
    print(f"Fixed Costs:                ${FIXED_COSTS:>9,.0f}")
    
    # Profit Summary
    print("\n" + "-"*70)
    print("PROFIT ANALYSIS")
    print("-"*70)
    print(f"Mean Profit:                ${stats['mean_profit']:>9,.2f}")
    print(f"Median Profit:              ${stats['median_profit']:>9,.2f}")
    print(f"Standard Deviation:         ${stats['std_profit']:>9,.2f}")
    print(f"Minimum Profit:             ${stats['min_profit']:>9,.2f}")
    print(f"Maximum Profit:             ${stats['max_profit']:>9,.2f}")
    
    # Percentiles
    print("\n" + "-"*70)
    print("PROFIT PERCENTILES (Confidence Intervals)")
    print("-"*70)
    print(f"5th Percentile (Worst 5%):  ${stats['p5_profit']:>9,.2f}")
    print(f"25th Percentile:            ${stats['p25_profit']:>9,.2f}")
    print(f"75th Percentile:            ${stats['p75_profit']:>9,.2f}")
    print(f"95th Percentile (Best 5%):  ${stats['p95_profit']:>9,.2f}")
    print(f"\nProfit Range (5-95):        ${stats['p95_profit'] - stats['p5_profit']:>9,.2f}")
    
    # Risk Analysis
    print("\n" + "-"*70)
    print("RISK ANALYSIS")
    print("-"*70)
    print(f"Probability of Loss:        {risk['prob_loss']:>10.2%}")
    print(f"Probability of Profit > $5K: {risk['prob_profit_5k']:>8.2%}")
    print(f"Probability of Profit > $10K: {risk['prob_profit_10k']:>7.2%}")
    
    # Financial Breakdown
    print("\n" + "-"*70)
    print("AVERAGE FINANCIAL METRICS")
    print("-"*70)
    print(f"Average Sales Volume:       {stats['mean_sales_volume']:>10,.0f} units")
    print(f"Average Unit Cost:          ${stats['mean_product_cost']:>9,.2f}/unit")
    print(f"Average Revenue:            ${stats['mean_revenue']:>9,.2f}")
    print(f"Average Variable Costs:     ${stats['mean_variable_costs']:>9,.2f}")
    print(f"Fixed Costs:                ${FIXED_COSTS:>9,.2f}")
    print(f"Average Total Costs:        ${stats['mean_total_costs']:>9,.2f}")
    print(f"Average Profit:             ${stats['mean_profit']:>9,.2f}")
    
    # Decision Support
    print("\n" + "-"*70)
    print("DECISION SUPPORT")
    print("-"*70)
    if risk['prob_loss'] < 0.05:
        print(f"✓ RISK LEVEL: LOW - Less than 5% chance of loss")
    elif risk['prob_loss'] < 0.15:
        print(f"⚠ RISK LEVEL: MODERATE - {risk['prob_loss']:.1%} chance of loss")
    else:
        print(f"✗ RISK LEVEL: HIGH - {risk['prob_loss']:.1%} chance of loss")
    
    if stats['mean_profit'] > 0:
        margin = (stats['mean_profit'] / stats['mean_revenue']) * 100
        print(f"✓ PROFITABILITY: Positive expected profit (${stats['mean_profit']:,.0f})")
        print(f"  Average profit margin: {margin:.1f}%")
    else:
        print(f"✗ PROFITABILITY: Negative expected profit (${stats['mean_profit']:,.0f})")
    
    print(f"✓ UPSIDE POTENTIAL: Best 5% of outcomes = ${stats['p95_profit']:,.0f}")
    print(f"✓ DOWNSIDE RISK: Worst 5% of outcomes = ${stats['p5_profit']:,.0f}")
    
    print("\n" + "="*70 + "\n")


def print_scenario_comparison(base_results):
    """Show impact of changing key parameters"""
    
    print("\n" + "="*70)
    print("SCENARIO ANALYSIS - HOW CHANGES AFFECT PROFIT")
    print("="*70)
    
    base_mean = base_results['stats']['mean_profit']
    
    print(f"\nBASE CASE Mean Profit: ${base_mean:,.2f}\n")
    
    # Scenario 1: Reduce fixed costs
    print("-"*70)
    print("SCENARIO 1: Reduce fixed costs from $5,000 to $3,000")
    print("-"*70)
    original_fixed = FIXED_COSTS
    # Simulate the impact
    reduced_profits = base_results['profits'] - (original_fixed - 3000)
    new_mean = reduced_profits.mean()
    improvement = new_mean - base_mean
    print(f"New Mean Profit:        ${new_mean:,.2f}")
    print(f"Improvement:            ${improvement:,.2f} ({(improvement/base_mean)*100:+.1f}%)")
    print(f"New Prob of Loss:       {(reduced_profits < 0).sum() / SALES_SIMULATIONS:.2%}")
    
    # Scenario 2: Tighter cost control
    print("\n" + "-"*70)
    print("SCENARIO 2: Tighter cost control ($18-$22 instead of $15-$25)")
    print("-"*70)
    narrower_costs = np.random.normal(COST_AVG, 1.0, size=SALES_SIMULATIONS)
    narrower_costs = np.clip(narrower_costs, 18, 22)
    narrower_profits = base_results['revenues'] - (base_results['sales_volumes'] * narrower_costs + FIXED_COSTS)
    new_mean = narrower_profits.mean()
    improvement = new_mean - base_mean
    print(f"New Mean Profit:        ${new_mean:,.2f}")
    print(f"Improvement:            ${improvement:,.2f} ({(improvement/base_mean)*100:+.1f}%)")
    print(f"New Std Deviation:      ${narrower_profits.std():,.2f} (was ${base_results['profits'].std():,.2f})")
    print(f"Profits More Stable:    Yes - lower uncertainty")
    
    # Scenario 3: Higher sales
    print("\n" + "-"*70)
    print("SCENARIO 3: Better market conditions (200-600 units instead of 100-500)")
    print("-"*70)
    higher_sales = np.random.triangular(200, 350, 600, size=SALES_SIMULATIONS)
    higher_profits = higher_sales * SALES_PRICE - (higher_sales * base_results['product_costs'] + FIXED_COSTS)
    new_mean = higher_profits.mean()
    improvement = new_mean - base_mean
    print(f"New Mean Profit:        ${new_mean:,.2f}")
    print(f"Improvement:            ${improvement:,.2f} ({(improvement/base_mean)*100:+.1f}%)")
    print(f"Best 5% outcome:        ${np.percentile(higher_profits, 95):,.2f} (was ${base_results['stats']['p95_profit']:,.2f})")
    
    # Scenario 4: Price increase
    print("\n" + "-"*70)
    print("SCENARIO 4: Increase selling price from $50 to $55/unit")
    print("-"*70)
    higher_price_profits = base_results['sales_volumes'] * 55 - base_results['total_costs']
    new_mean = higher_price_profits.mean()
    improvement = new_mean - base_mean
    print(f"New Mean Profit:        ${new_mean:,.2f}")
    print(f"Improvement:            ${improvement:,.2f} ({(improvement/base_mean)*100:+.1f}%)")
    print(f"New Prob of Loss:       {(higher_price_profits < 0).sum() / SALES_SIMULATIONS:.2%}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Which scenario creates the biggest improvement?")
    print("Focus your efforts there!")
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("STARTING COST & SALES MONTE CARLO SIMULATION...")
    print("="*70)
    print(f"Running {SALES_SIMULATIONS:,} simulations...")
    
    # Run the main simulation
    results = simulate_cost_and_sales()
    
    # Print detailed results
    print_detailed_results(results)
    
    # Print scenario analysis
    print_scenario_comparison(results)
    
    # Export to CSV for further analysis
    print("\n" + "-"*70)
    print("EXPORTING RESULTS")
    print("-"*70)
    
    # Create results array for export
    export_data = np.column_stack([
        results['sales_volumes'],
        results['product_costs'],
        results['revenues'],
        results['total_costs'],
        results['profits']
    ])
    
    # Save to CSV
    np.savetxt(
        'monte_carlo_results.csv',
        export_data,
        delimiter=',',
        header='Sales_Volume,Product_Cost,Revenue,Total_Costs,Profit',
        comments='',
        fmt='%.2f'
    )
    
    print("✓ Results exported to: monte_carlo_results.csv")
    print(f"  Contains {SALES_SIMULATIONS:,} rows of simulation data")
    print("  Use in Excel or other tools for further analysis")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70 + "\n")
    
    # Summary for quick reference
    print("QUICK REFERENCE:")
    print(f"Expected Profit:  ${results['stats']['mean_profit']:>10,.2f}")
    print(f"Best Case (95%):  ${results['stats']['p95_profit']:>10,.2f}")
    print(f"Worst Case (5%):  ${results['stats']['p5_profit']:>10,.2f}")
    print(f"Risk of Loss:     {results['risk_metrics']['prob_loss']:>10.2%}\n")