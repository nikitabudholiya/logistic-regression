# Logistic Regression: A Simple Story

A short, beginner-friendly guide to logistic regression, the sigmoid function, and the loss function.

---

## 1. Label and Class

These are basically the same thing. Take a dataset of students:

| Hours studied | Result |
|---|---|
| 1 | Fail |
| 2 | Fail |
| 6 | Pass |
| 8 | Pass |

- **Class**: the categories (*Fail* and *Pass*), written as **0** and **1**.
- **Label (y)**: the true answer for each data point, already known in the data (0 or 1).

---

## 2. Where it starts: the Perceptron Trick

We have a line that separates the two classes. Each point is asked: "Are you on the correct side?"

- Correctly classified: the line stays put.
- Misclassified: the point pulls the line toward itself.

**Drawbacks:**
1. Only mistakes move the line, so it stops at the first line that works, not the best one.
2. Everything is yes/no. A point slightly wrong and a point very wrong are treated the same.

---

## 3. The Fix: everyone gets a vote

Logistic regression improves on this:

- **Misclassified points pull** the line toward them.
- **Correctly classified points push** the line away.
- **The strength of the pull or push depends on the distance** of the point from the line.

The line keeps adjusting until it sits in a good position, not just any position that separates the classes.

---

## 4. The Story, Step by Step

**Goal:** a new student studied 5 hours. Will they pass?

### Step 1: Draw a line
The line separates fail students from pass students.

### Step 2: Measure distance (z)
```
z = w·x + b
```
- Far on the pass side: z is a large positive number
- Far on the fail side: z is a large negative number
- On the line: z = 0

### Step 3: Convert distance to probability (Sigmoid)
A raw distance is hard to interpret, so the sigmoid turns it into a probability between 0 and 1:

```
ŷ = σ(z) = 1 / (1 + e^(-z))
```

| z (distance) | σ(z) (probability of class 1) |
|---|---|
| -5 | 0.007 |
| -2 | 0.12 |
| 0 | 0.50 |
| +2 | 0.88 |
| +5 | 0.993 |

- z > 0 gives σ(z) > 0.5, so predict **class 1**
- z < 0 gives σ(z) < 0.5, so predict **class 0**
- z = 0 gives σ(z) = 0.5, so the point is exactly on the line

> The sigmoid does **not** say "correct" or "incorrect". It says which class the model believes the point belongs to. Whether that belief is right is decided by comparing with the true label y.

The sigmoid is an S-shaped curve: steep near the line (confidence changes fast) and flat far away (already sure). It is the perceptron's hard step function with the corners smoothed out.

### Step 4: Check against the truth (Loss function)
The loss function is a score for how wrong the model was. Low is good, high is bad.

```
Loss = -[ y·log(ŷ) + (1 - y)·log(1 - ŷ) ]
```

This is called **log loss** (binary cross-entropy). It only does one thing:

- If true y = 1: loss = -log(ŷ). Small if ŷ is near 1, huge if ŷ is near 0.
- If true y = 0: loss = -log(1 - ŷ). Small if ŷ is near 0, huge if ŷ is near 1.

Confidently wrong predictions are punished heavily. Confidently right predictions cost almost nothing.

| True label | ŷ | Verdict |
|---|---|---|
| 1 | 0.9 | Correct, confident (tiny loss) |
| 1 | 0.2 | Wrong (large loss, pulls the line hard) |
| 0 | 0.1 | Correct, confident (tiny loss) |
| 0 | 0.8 | Wrong (large loss, pulls the line hard) |

### Step 5: Fix the line (Learning)
Each point contributes an update based on its error:

```
error = y - ŷ
w = w + learning_rate × (y - ŷ) × x
```

- Misclassified and far away: error is about ±1, so a strong pull
- Correct and far away: error is about 0, so a tiny push
- Near the line (ŷ ≈ 0.5): moderate effect

This is the same formula as the perceptron trick. The only difference is that ŷ is now a smooth probability instead of a hard 0 or 1, so every point contributes. Gradient descent on the log loss produces exactly this update.

Repeat over many rounds and the line settles where the total pull and push balance.

---

## 5. The Full Chain

```
point → z (distance from line) → σ(z) (probability)
      → compare with true label y → loss / error (y - ŷ)
      → pull or push the line
```

To make a final prediction, apply a threshold (usually 0.5, which corresponds to z = 0, the line itself).

---

## 6. One-Line Summary

**Draw a line, measure distance, turn distance into a probability, score how wrong it was, and move the line to reduce that score.**

---

## Quick Glossary

| Term | Meaning |
|---|---|
| Class | A category (0 or 1) |
| Label (y) | The true class of a data point |
| z | Signed distance of a point from the line (w·x + b) |
| Sigmoid | Function that converts z into a probability |
| ŷ | Predicted probability of class 1 |
| Loss function | A score of how wrong the prediction was |
| Log loss | The specific loss used in logistic regression |
| Gradient descent | Repeatedly adjusting w and b to reduce the loss |
