"""
PROBABILITY DISTRIBUTIONS FUNDAMENTALS
Complete guide with theory and interactive code

Learn:
- What distributions are
- When to use each one
- How they work mathematically
- Real-world examples
- How to use them in Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# PART 1: FOUNDATIONS
# ============================================================================

"""
WHAT IS A PROBABILITY DISTRIBUTION?

A probability distribution describes the likelihood of different outcomes.

Instead of: "There's a 50% chance it rains" (one outcome)
Distribution says: "10% chance 0mm rain, 20% chance 1mm, 40% chance 5mm, etc."
                   Shows FULL range of possibilities

TWO MAIN TYPES:
1. Discrete - Can only take specific values (1, 2, 3... not 2.5)
   Example: Number of coin flips before getting heads

2. Continuous - Can take any value in a range (1.5, 2.75, 3.14159...)
   Example: Height of people (5.8 feet, 5.81234... feet)
"""

# ============================================================================
# PART 2: DISCRETE DISTRIBUTIONS
# ============================================================================

print("\n" + "="*70)
print("DISCRETE DISTRIBUTIONS (Whole numbers only)")
print("="*70)

# ============================================================================
# 2.1: BINOMIAL DISTRIBUTION
# ============================================================================

print("\n" + "-"*70)
print("1. BINOMIAL DISTRIBUTION")
print("-"*70)

"""
WHAT IS IT?
Counts the number of successes in a fixed number of trials.

KEY REQUIREMENTS:
- Fixed number of trials (n)
- Each trial has two outcomes: success or failure
- Probability of success is same every trial (p)
- Trials are independent

FORMULA:
P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

WHERE:
- n = number of trials
- k = number of successes we want
- p = probability of success on each trial
- C(n,k) = "n choose k" = ways to arrange k successes in n trials

REAL WORLD EXAMPLES:
✓ Flipping coin 100 times, count heads (n=100, p=0.5)
✓ Product defects: 1000 items, 2% defect rate (n=1000, p=0.02)
✓ Email clicks: Send 5000 emails, 3% click rate (n=5000, p=0.03)
✓ Customer conversions: 100 visitors, 5% convert (n=100, p=0.05)
"""

# Example 1: Coin flips
print("\nEXAMPLE 1: Coin Flips")
print("-" * 40)

n_trials = 100          # Flip coin 100 times
p_success = 0.5         # 50% chance of heads
n_simulations = 10000

# Simulate: How many heads in 100 flips? Do this 10,000 times
heads_counts = np.random.binomial(n=n_trials, p=p_success, size=n_simulations)

print(f"Question: Flip coin 100 times, how many heads?")
print(f"Running 10,000 simulations...\n")
print(f"Average heads: {heads_counts.mean():.1f}")
print(f"Min heads: {heads_counts.min()}")
print(f"Max heads: {heads_counts.max()}")
print(f"Std Dev: {heads_counts.std():.1f}")
print(f"Most common: {np.bincount(heads_counts).argmax()} heads")
print(f"\nProbabilities:")
print(f"  Prob of 50 heads exactly: {(heads_counts == 50).sum() / n_simulations:.2%}")
print(f"  Prob of 45-55 heads: {((heads_counts >= 45) & (heads_counts <= 55)).sum() / n_simulations:.2%}")
print(f"  Prob of 40+ heads: {(heads_counts >= 40).sum() / n_simulations:.2%}")

# Example 2: Product defects
print("\n\nEXAMPLE 2: Product Defects")
print("-" * 40)

batch_size = 1000       # 1000 items in a batch
defect_rate = 0.02      # 2% defect rate
n_simulations = 10000

# Simulate: How many defects in a batch? Do this 10,000 times
defects = np.random.binomial(n=batch_size, p=defect_rate, size=n_simulations)

print(f"Question: In a batch of 1000 items with 2% defect rate,")
print(f"how many defects will we have? (10,000 batches)\n")
print(f"Average defects per batch: {defects.mean():.1f}")
print(f"Min defects: {defects.min()}")
print(f"Max defects: {defects.max()}")
print(f"\nProbabilities:")
print(f"  Prob of exactly 20 defects: {(defects == 20).sum() / n_simulations:.2%}")
print(f"  Prob of fewer than 15 defects: {(defects < 15).sum() / n_simulations:.2%}")
print(f"  Prob of more than 30 defects: {(defects > 30).sum() / n_simulations:.2%}")

# Example 3: Email campaign
print("\n\nEXAMPLE 3: Email Marketing Click-through")
print("-" * 40)

emails_sent = 5000      # Send 5000 emails
click_rate = 0.03       # 3% click rate
n_simulations = 10000

clicks = np.random.binomial(n=emails_sent, p=click_rate, size=n_simulations)

print(f"Question: Send 5000 emails with 3% click rate,")
print(f"how many clicks? (10,000 campaigns)\n")
print(f"Average clicks: {clicks.mean():.1f}")
print(f"Range: {clicks.min()} - {clicks.max()} clicks")
print(f"Std Dev: {clicks.std():.1f}")
print(f"\nProbabilities:")
print(f"  Prob of 150+ clicks (good): {(clicks >= 150).sum() / n_simulations:.2%}")
print(f"  Prob of exactly 150 clicks: {(clicks == 150).sum() / n_simulations:.2%}")
print(f"  Prob of fewer than 130 clicks (bad): {(clicks < 130).sum() / n_simulations:.2%}")

# ============================================================================
# 2.2: POISSON DISTRIBUTION
# ============================================================================

print("\n" + "-"*70)
print("2. POISSON DISTRIBUTION")
print("-"*70)

"""
WHAT IS IT?
Counts the number of events occurring in a FIXED TIME PERIOD or SPACE.

KEY IDEA:
- You don't know exactly when events happen
- But you know the AVERAGE rate
- Poisson tells you distribution around that average

KEY REQUIREMENT:
- Events are independent (one doesn't affect another)
- Average rate is constant

PARAMETER:
- λ (lambda) = average number of events in time period
  (e.g., 3 customer arrivals per hour)

FORMULA:
P(X=k) = (e^-λ * λ^k) / k!

WHERE:
- e ≈ 2.718 (Euler's number)
- k! = k factorial (k × (k-1) × (k-2)...)
- λ = average rate

REAL WORLD EXAMPLES:
✓ Customer arrivals: 10 customers per hour on average
✓ Server requests: 100 requests per minute on average
✓ Defects: 2 defects per 1000 items on average
✓ Accidents: 4 car accidents per day at intersection
✓ Calls: 5 support calls per hour on average
✓ Website visitors: 1000 visitors per day on average

WHEN TO USE BINOMIAL vs POISSON:
- Binomial: Fixed number of trials (n is known)
  Example: Out of 100 people, how many will buy?
  
- Poisson: Unknown number of events in time/space
  Example: During 1 hour, how many customers will arrive?
"""

# Example 1: Customer arrivals
print("\nEXAMPLE 1: Customer Arrivals")
print("-" * 40)

lambda_customers = 10   # 10 customers per hour on average
hours = 8               # Working 8 hours
n_simulations = 10000

# Simulate: How many customers arrive? Do this 10,000 times
customers_per_hour = np.random.poisson(lam=lambda_customers, size=n_simulations)

print(f"Question: Store gets 10 customers/hour on average.")
print(f"How many customers in a typical hour? (10,000 simulations)\n")
print(f"Average: {customers_per_hour.mean():.1f} customers")
print(f"Min: {customers_per_hour.min()}")
print(f"Max: {customers_per_hour.max()}")
print(f"Std Dev: {customers_per_hour.std():.1f}")
print(f"\nProbabilities:")
print(f"  Prob of exactly 10 customers: {(customers_per_hour == 10).sum() / n_simulations:.2%}")
print(f"  Prob of 8-12 customers (normal): {((customers_per_hour >= 8) & (customers_per_hour <= 12)).sum() / n_simulations:.2%}")
print(f"  Prob of 15+ customers (busy): {(customers_per_hour >= 15).sum() / n_simulations:.2%}")
print(f"  Prob of 5 or fewer (slow): {(customers_per_hour <= 5).sum() / n_simulations:.2%}")

# Example 2: Website errors
print("\n\nEXAMPLE 2: Website Errors")
print("-" * 40)

lambda_errors = 2       # 2 errors per hour on average
n_simulations = 10000

errors_per_hour = np.random.poisson(lam=lambda_errors, size=n_simulations)

print(f"Question: Website has 2 errors/hour on average.")
print(f"What's the chance of 5+ errors? (10,000 hours)\n")
print(f"Average errors: {errors_per_hour.mean():.1f}")
print(f"Prob of 0 errors (perfect hour): {(errors_per_hour == 0).sum() / n_simulations:.2%}")
print(f"Prob of 1-3 errors (acceptable): {((errors_per_hour >= 1) & (errors_per_hour <= 3)).sum() / n_simulations:.2%}")
print(f"Prob of 5+ errors (bad): {(errors_per_hour >= 5).sum() / n_simulations:.2%}")

# Example 3: Support tickets
print("\n\nEXAMPLE 3: Support Tickets Received")
print("-" * 40)

lambda_tickets = 5      # 5 tickets per hour
n_simulations = 10000

tickets_per_hour = np.random.poisson(lam=lambda_tickets, size=n_simulations)

print(f"Question: Support team gets 5 tickets/hour average.")
print(f"How many hours will have 8+ tickets? (10,000 hours)\n")
print(f"Average tickets: {tickets_per_hour.mean():.1f}")
print(f"Hours with 8+ tickets: {(tickets_per_hour >= 8).sum() / n_simulations:.2%}")
print(f"Need to staff for overflow: {(tickets_per_hour >= 10).sum() / n_simulations:.2%}")

# ============================================================================
# PART 3: CONTINUOUS DISTRIBUTIONS
# ============================================================================

print("\n" + "="*70)
print("CONTINUOUS DISTRIBUTIONS (Any decimal value)")
print("="*70)

# ============================================================================
# 3.1: NORMAL DISTRIBUTION (Gaussian)
# ============================================================================

print("\n" + "-"*70)
print("3. NORMAL DISTRIBUTION (Bell Curve)")
print("-"*70)

"""
WHAT IS IT?
The most famous distribution. Bell-shaped curve centered around the mean.

WHY SO COMMON?
"Central Limit Theorem" = When you add many random things together,
the result is always normally distributed!

KEY PARAMETERS:
- μ (mu) = mean (center of bell)
- σ (sigma) = standard deviation (width of bell)

FORMULA:
Too complex! But you don't need it - use numpy.random.normal()

REAL WORLD EXAMPLES:
✓ Heights of people (mean 5'9", most cluster near center)
✓ Test scores (mean 100, most students score 85-115)
✓ Product weights (mean 500g, some 495g, some 505g)
✓ Daily temperatures
✓ Measurement errors
✓ IQ scores
✓ Stock returns (approximately)

SPECIAL FEATURE - 68-95-99.7 RULE:
- 68% of values fall within 1 std dev of mean
- 95% within 2 std devs
- 99.7% within 3 std devs

Example: Heights with mean 70", std dev 3"
- 68% of people are 67-73" (70±3)
- 95% are 64-76" (70±6)
- 99.7% are 61-79" (70±9)
"""

# Example 1: Heights
print("\nEXAMPLE 1: Human Heights")
print("-" * 40)

mean_height = 70        # 70 inches (5'10")
std_dev = 3             # 3 inch variation
n_simulations = 10000

heights = np.random.normal(loc=mean_height, scale=std_dev, size=n_simulations)

print(f"Question: Population has mean height 70\", std dev 3\"")
print(f"What are typical heights? (10,000 people)\n")
print(f"Average height: {heights.mean():.2f}\"")
print(f"Std Dev: {heights.std():.2f}\"")
print(f"Min: {heights.min():.2f}\"")
print(f"Max: {heights.max():.2f}\"")
print(f"\nPercentiles:")
print(f"  5th (shortest 5%): {np.percentile(heights, 5):.2f}\"")
print(f"  50th (median): {np.percentile(heights, 50):.2f}\"")
print(f"  95th (tallest 5%): {np.percentile(heights, 95):.2f}\"")
print(f"\n68-95-99.7 Rule Check:")
print(f"  % within 1 std dev (67-73\"): {((heights >= 67) & (heights <= 73)).sum() / n_simulations:.1%}")
print(f"  % within 2 std devs (64-76\"): {((heights >= 64) & (heights <= 76)).sum() / n_simulations:.1%}")
print(f"  % within 3 std devs (61-79\"): {((heights >= 61) & (heights <= 79)).sum() / n_simulations:.1%}")

# Example 2: Test scores
print("\n\nEXAMPLE 2: Test Scores")
print("-" * 40)

mean_score = 100        # Average score
std_dev = 15            # Standard deviation
n_simulations = 10000

scores = np.random.normal(loc=mean_score, scale=std_dev, size=n_simulations)

print(f"Question: Test scores normally distributed, mean 100, std dev 15")
print(f"What grades will students get? (10,000 students)\n")
print(f"Average score: {scores.mean():.1f}")
print(f"Std Dev: {scores.std():.1f}")
print(f"\nGrade Distribution:")
print(f"  Prob of A (90+): {(scores >= 90).sum() / n_simulations:.2%}")
print(f"  Prob of B (80-89): {((scores >= 80) & (scores < 90)).sum() / n_simulations:.2%}")
print(f"  Prob of C (70-79): {((scores >= 70) & (scores < 80)).sum() / n_simulations:.2%}")
print(f"  Prob of D (60-69): {((scores >= 60) & (scores < 70)).sum() / n_simulations:.2%}")
print(f"  Prob of F (< 60): {(scores < 60).sum() / n_simulations:.2%}")

# Example 3: Product weight
print("\n\nEXAMPLE 3: Product Manufacturing")
print("-" * 40)

target_weight = 500     # Target 500g
tolerance = 10          # Std dev 10g
n_simulations = 10000

weights = np.random.normal(loc=target_weight, scale=tolerance, size=n_simulations)

print(f"Question: Package target is 500g ± 10g std dev")
print(f"How many are within spec (490-510g)? (10,000 packages)\n")
print(f"Average weight: {weights.mean():.2f}g")
print(f"Std Dev: {weights.std():.2f}g")
within_spec = ((weights >= 490) & (weights <= 510)).sum() / n_simulations
print(f"\nWithin spec (490-510g): {within_spec:.2%}")
print(f"Need to reject: {(1-within_spec):.2%}")

# ============================================================================
# 3.2: GEOMETRIC DISTRIBUTION
# ============================================================================

print("\n" + "-"*70)
print("4. GEOMETRIC DISTRIBUTION")
print("-"*70)

"""
WHAT IS IT?
Counts how many trials until you get the FIRST SUCCESS.

KEY IDEA:
- You keep trying until you succeed
- Each attempt has same success probability
- How many attempts until first success?

PARAMETER:
- p = probability of success on each trial (0 to 1)

FORMULA:
P(X=k) = (1-p)^(k-1) * p

WHERE:
- k = number of trials until first success
- p = probability of success each trial
- (1-p)^(k-1) = probability of k-1 failures

REAL WORLD EXAMPLES:
✓ Coin flips until first heads (p=0.5)
✓ Job interviews until first offer (p=0.2)
✓ Sales calls until first deal (p=0.1)
✓ Shooting until first successful shot (p=0.8)
✓ Lottery tickets until you win (p=very small)
✓ Customer service calls until resolved (p=0.6)

SPECIAL PROPERTY:
Average number of trials = 1/p
- If 10% success rate, expect 1/0.1 = 10 trials
- If 50% success rate, expect 1/0.5 = 2 trials
- If 1% success rate, expect 1/0.01 = 100 trials
"""

# Example 1: Coin flips until heads
print("\nEXAMPLE 1: Coin Flips Until First Heads")
print("-" * 40)

p_success = 0.5         # 50% chance heads
n_simulations = 10000

# How many flips until first heads?
flips_until_heads = np.random.geometric(p=p_success, size=n_simulations)

print(f"Question: Flip coin until get heads. How many flips? (10,000 trials)\n")
print(f"Average flips needed: {flips_until_heads.mean():.1f}")
print(f"Min flips: {flips_until_heads.min()}")
print(f"Max flips: {flips_until_heads.max()}")
print(f"Median (50%): {np.median(flips_until_heads):.0f}")
print(f"\nProbabilities:")
print(f"  Prob get heads on 1st flip: {(flips_until_heads == 1).sum() / n_simulations:.2%}")
print(f"  Prob get heads by 3rd flip: {(flips_until_heads <= 3).sum() / n_simulations:.2%}")
print(f"  Prob need 10+ flips: {(flips_until_heads >= 10).sum() / n_simulations:.2%}")

# Example 2: Sales calls until first deal
print("\n\nEXAMPLE 2: Sales Calls Until First Deal")
print("-" * 40)

p_sale = 0.1            # 10% of calls result in sale
n_simulations = 10000

calls_until_sale = np.random.geometric(p=p_sale, size=n_simulations)

print(f"Question: Sales rep has 10% close rate.")
print(f"How many calls until first sale? (10,000 reps)\n")
print(f"Average calls needed: {calls_until_sale.mean():.1f}")
print(f"Min calls: {calls_until_sale.min()}")
print(f"Max calls: {calls_until_sale.max()}")
print(f"\nProbabilities:")
print(f"  Prob of 1st call closes: {(calls_until_sale == 1).sum() / n_simulations:.2%}")
print(f"  Prob within 5 calls: {(calls_until_sale <= 5).sum() / n_simulations:.2%}")
print(f"  Prob within 10 calls: {(calls_until_sale <= 10).sum() / n_simulations:.2%}")
print(f"  Prob need 20+ calls: {(calls_until_sale >= 20).sum() / n_simulations:.2%}")

# Example 3: Job interviews until offer
print("\n\nEXAMPLE 3: Job Interviews Until Offer")
print("-" * 40)

p_offer = 0.2           # 20% of interviews result in offer
n_simulations = 10000

interviews_until_offer = np.random.geometric(p=p_offer, size=n_simulations)

print(f"Question: Have 20% interview success rate.")
print(f"How many interviews until 1st offer? (10,000 people)\n")
print(f"Average interviews: {interviews_until_offer.mean():.1f}")
print(f"Range: {interviews_until_offer.min()} - {interviews_until_offer.max()}")
print(f"Prob of offer on 1st interview: {(interviews_until_offer == 1).sum() / n_simulations:.2%}")
print(f"Prob of 1st offer within 3 interviews: {(interviews_until_offer <= 3).sum() / n_simulations:.2%}")

# ============================================================================
# PART 4: COMPARISON & WHEN TO USE
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: WHEN TO USE EACH DISTRIBUTION")
print("="*70)

comparison = """
┌─────────────┬──────────────┬─────────────────────────────────────┐
│ Distribution│ Type        │ When to Use                         │
├─────────────┼──────────────┼─────────────────────────────────────┤
│ BINOMIAL    │ Discrete    │ Fixed # of trials, 2 outcomes      │
│             │             │ Example: 100 people, how many buy? │
│             │             │                                    │
│ POISSON     │ Discrete    │ # of events in time/space period   │
│             │             │ Example: customers per hour        │
│             │             │                                    │
│ GEOMETRIC   │ Discrete    │ Trials until 1st success          │
│             │             │ Example: calls until sale closes   │
│             │             │                                    │
│ NORMAL      │ Continuous  │ Natural variation around average   │
│             │             │ Example: heights, test scores      │
│             │             │ (Most common distribution!)        │
└─────────────┴──────────────┴─────────────────────────────────────┘

KEY PROPERTIES:

Binomial:
  - Has n (number of trials) and p (success probability)
  - Mean = n*p
  - Shape depends on p and n
  - Example: np.random.binomial(n=100, p=0.5, size=1000)

Poisson:
  - Has λ (lambda) = average rate
  - Mean = λ
  - Good for rare events
  - Example: np.random.poisson(lam=5, size=1000)

Geometric:
  - Has p (success probability)
  - Mean = 1/p
  - Always skewed (most values near 1, some very large)
  - Example: np.random.geometric(p=0.1, size=1000)

Normal:
  - Has μ (mean) and σ (standard deviation)
  - Bell-shaped, symmetric
  - 68-95-99.7 rule
  - Example: np.random.normal(loc=0, scale=1, size=1000)
"""

print(comparison)

# ============================================================================
# PART 5: PRACTICAL EXAMPLES FOR MONTE CARLO
# ============================================================================

print("\n" + "="*70)
print("PRACTICAL: USING DISTRIBUTIONS IN MONTE CARLO")
print("="*70)

print("""
EXAMPLE: Ecommerce Product Launch

Question: How much revenue will we make?

Sources of Uncertainty:
1. Number of visitors (POISSON - events over time)
   - Estimate: 1000 visitors per day on average
   
2. Conversion rate (BINOMIAL - success/failure)
   - Estimate: 2% of visitors buy
   
3. Order amount (NORMAL - natural variation)
   - Estimate: $50 average, $15 std dev

Simulation:
""")

# Setup
np.random.seed(42)
n_days = 30
n_simulations = 10000

daily_revenue = []

for sim in range(n_simulations):
    # Day-by-day revenue for 30 days
    daily_totals = []
    
    for day in range(n_days):
        # 1. How many visitors today? (Poisson)
        visitors = np.random.poisson(lam=1000)
        
        # 2. How many convert? (Binomial)
        purchases = np.random.binomial(n=visitors, p=0.02)
        
        # 3. What's average order amount? (Normal)
        if purchases > 0:
            order_amounts = np.random.normal(loc=50, scale=15, size=purchases)
            daily_revenue_total = order_amounts.sum()
        else:
            daily_revenue_total = 0
        
        daily_totals.append(daily_revenue_total)
    
    # Total revenue for 30 days
    total_revenue = sum(daily_totals)
    daily_revenue.append(total_revenue)

daily_revenue = np.array(daily_revenue)

print(f"30-Day Revenue Forecast ({n_simulations:,} simulations):\n")
print(f"Average revenue: ${daily_revenue.mean():,.2f}")
print(f"Median revenue: ${np.median(daily_revenue):,.2f}")
print(f"Std Dev: ${daily_revenue.std():,.2f}")
print(f"Best 5%: ${np.percentile(daily_revenue, 95):,.2f}")
print(f"Worst 5%: ${np.percentile(daily_revenue, 5):,.2f}")

print("\n" + "="*70)
print("CREATING VISUALIZATIONS...")
print("="*70 + "\n")

# ============================================================================
# PART 6: VISUALIZATIONS
# ============================================================================

# Create a large figure with multiple subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Probability Distributions Visualization', fontsize=16, fontweight='bold')

# ============================================================================
# 1. BINOMIAL DISTRIBUTION
# ============================================================================

ax = axes[0, 0]
binomial_data = np.random.binomial(n=100, p=0.5, size=10000)
ax.hist(binomial_data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(binomial_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {binomial_data.mean():.1f}')
ax.set_title('Binomial Distribution\n(100 coin flips, p=0.5)', fontweight='bold')
ax.set_xlabel('Number of Heads')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# ============================================================================
# 2. POISSON DISTRIBUTION
# ============================================================================

ax = axes[0, 1]
poisson_data = np.random.poisson(lam=10, size=10000)
ax.hist(poisson_data, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(poisson_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {poisson_data.mean():.1f}')
ax.set_title('Poisson Distribution\n(λ=10, events per time period)', fontweight='bold')
ax.set_xlabel('Number of Events')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# ============================================================================
# 3. GEOMETRIC DISTRIBUTION
# ============================================================================

ax = axes[1, 0]
geometric_data = np.random.geometric(p=0.1, size=10000)
# Limit for visualization (most values are small)
geometric_data_clipped = geometric_data[geometric_data <= 50]
ax.hist(geometric_data_clipped, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax.axvline(geometric_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {geometric_data.mean():.1f}')
ax.set_title('Geometric Distribution\n(p=0.1, trials until success)', fontweight='bold')
ax.set_xlabel('Number of Trials')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# ============================================================================
# 4. NORMAL DISTRIBUTION
# ============================================================================

ax = axes[1, 1]
normal_data = np.random.normal(loc=70, scale=3, size=10000)
ax.hist(normal_data, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7, density=True)
# Add a bell curve overlay
from scipy import stats
x = np.linspace(normal_data.min(), normal_data.max(), 100)
ax.plot(x, stats.norm.pdf(x, loc=70, scale=3), 'r-', linewidth=2, label='Bell Curve')
ax.axvline(normal_data.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {normal_data.mean():.1f}')
ax.set_title('Normal Distribution\n(μ=70, σ=3, heights)', fontweight='bold')
ax.set_xlabel('Height (inches)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# ============================================================================
# 5. COMPARISON: MULTIPLE BINOMIAL PROBABILITIES
# ============================================================================

ax = axes[2, 0]
# Show how binomial changes with different p values
colors = ['blue', 'green', 'red', 'orange']
for idx, p in enumerate([0.25, 0.5, 0.75, 0.9]):
    data = np.random.binomial(n=20, p=p, size=10000)
    ax.hist(data, bins=20, alpha=0.5, label=f'p={p}', color=colors[idx], edgecolor='black')
ax.set_title('Binomial Distribution\n(Different Probabilities, n=20)', fontweight='bold')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# ============================================================================
# 6. COMPARISON: MULTIPLE POISSON RATES
# ============================================================================

ax = axes[2, 1]
# Show how poisson changes with different lambda values
colors = ['blue', 'green', 'red', 'orange']
for idx, lam in enumerate([2, 5, 10, 15]):
    data = np.random.poisson(lam=lam, size=10000)
    ax.hist(data, bins=30, alpha=0.5, label=f'λ={lam}', color=colors[idx], edgecolor='black')
ax.set_title('Poisson Distribution\n(Different Average Rates)', fontweight='bold')
ax.set_xlabel('Number of Events')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.savefig('probability_distributions_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as: probability_distributions_visualization.png")

plt.show()

print("\n" + "="*70)
print("SIMULATION COMPLETE!")
print("="*70 + "\n")