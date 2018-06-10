import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

public class OnIce {

  public static final double GOLD_REWARD = 100.0;
  public static final double PIT_REWARD = -150.0;
  public static final double DISCOUNT_FACTOR = 0.5;
  public static final double EXPLORE_PROB = 0.2;  // for Q-learning
  public static final double LEARNING_RATE = 0.1;
  public static final int ITERATIONS = 10000;
  public static final int MAX_MOVES = 1000;

  public static final String[] directionStrings = new String[]{"U", "R", "D", "L"};

  // Using a fixed random seed so that the behavior is a little
  // more reproducible across runs & students
  public static Random rng = new Random(2018);

  public static void main(String[] args) throws FileNotFoundException {
//    Scanner myScanner = new Scanner(System.in);
    Scanner myScanner = new Scanner(new File("Input3.txt"));
//    Scanner myScanner = new Scanner(new File("bigQ.txt"));
    Problem problem = new Problem(myScanner);
    Policy policy = problem.solve(ITERATIONS);
    if (policy == null) {
      System.err.println("No policy.  Invalid solution approach?");
    } else {
      System.out.println(policy);
    }
    if (args.length > 0 && args[0].equals("eval")) {
      System.out.println("Average utility per move: "
              + tryPolicy(policy, problem));
    }
  }

  private static class Problem {
    public String approach;
    public double[] moveProbs;
    public ArrayList<ArrayList<String>> map;

    // Format looks like
    // MDP    [approach to be used]
    // 0.7 0.2 0.1   [probability of going 1, 2, 3 spaces]
    // - - - - - - P - - - -   [space-delimited map rows]
    // - - G - - - - - P - -   [G is gold, P is pit]
    //
    // You can assume the maps are rectangular, although this isn't enforced
    // by this constructor.

    Problem(Scanner sc) {
      approach = sc.nextLine();
      String probsString = sc.nextLine();
      String[] probsStrings = probsString.split(" ");
      moveProbs = new double[probsStrings.length];
      for (int i = 0; i < probsStrings.length; i++) {
        try {
          moveProbs[i] = Double.parseDouble(probsStrings[i]);
        } catch (NumberFormatException e) {
          break;
        }
      }
      map = new ArrayList<ArrayList<String>>();
      while (sc.hasNextLine()) {
        String line = sc.nextLine();
        String[] squares = line.split(" ");
        ArrayList<String> row = new ArrayList<String>(Arrays.asList(squares));
        map.add(row);
      }
    }

    Policy solve(int iterations) {
      if (approach.equals("MDP")) {
        MDPSolver mdp = new MDPSolver(this);
        return mdp.solve(this, iterations);
      } else if (approach.equals("Q")) {
        QLearner q = new QLearner(this);
        return q.solve(this, iterations);
      }
      return null;
    }

  }

  private static class Policy {
    public String[][] bestActions;

    public Policy(Problem prob) {
      bestActions = new String[prob.map.size()][prob.map.get(0).size()];
    }

    public String toString() {
      String out = "";
      for (int r = 0; r < bestActions.length; r++) {
        for (int c = 0; c < bestActions[0].length; c++) {
          if (c != 0) {
            out += " ";
          }
          out += bestActions[r][c];
        }
        out += "\n";
      }
      return out;
    }
  }

  // Returns the average utility per move of the policy,
  // as measured from ITERATIONS random drops of an agent onto
  // empty spaces
  public static double tryPolicy(Policy policy, Problem prob) {
    int totalUtility = 0;
    int totalMoves = 0;
    for (int i = 0; i < ITERATIONS; i++) {
      // Random empty starting loc
      int row, col;
      do {
        row = rng.nextInt(prob.map.size());
        col = rng.nextInt(prob.map.get(0).size());
      } while (!prob.map.get(row).get(col).equals("-"));
      // Run until pit, gold, or MAX_MOVES timeout
      // (in case policy recommends driving into wall repeatedly,
      // for example)
      for (int moves = 0; moves < MAX_MOVES; moves++) {
        totalMoves++;
        String policyRec = policy.bestActions[row][col];
        // Determine how far we go in that direction
        int displacement = 1;
        double totalProb = 0;
        double moveSample = rng.nextDouble();
        for (int p = 0; p <= prob.moveProbs.length; p++) {
          totalProb += prob.moveProbs[p];
          if (moveSample <= totalProb) {
            displacement = p + 1;
            break;
          }
        }
        int new_row = row;
        int new_col = col;
        if (policyRec.equals("U")) {
          new_row -= displacement;
          if (new_row < 0) {
            new_row = 0;
          }
        } else if (policyRec.equals("R")) {
          new_col += displacement;
          if (new_col >= prob.map.get(0).size()) {
            new_col = prob.map.get(0).size() - 1;
          }
        } else if (policyRec.equals("D")) {
          new_row += displacement;
          if (new_row >= prob.map.size()) {
            new_row = prob.map.size() - 1;
          }
        } else if (policyRec.equals("L")) {
          new_col -= displacement;
          if (new_col < 0) {
            new_col = 0;
          }
        }
        row = new_row;
        col = new_col;
        if (prob.map.get(row).get(col).equals("G")) {
          totalUtility += GOLD_REWARD;
          // End the current trial
          break;
        } else if (prob.map.get(row).get(col).equals("P")) {
          totalUtility += PIT_REWARD;
          break;
        }
      }
    }

    return totalUtility / (double) totalMoves;
  }


  private static class MDPSolver {

    // We'll want easy access to the real rewards while iterating, so
    // we'll keep both of these around
    public double[][] utilities;
    public double[][] rewards;

    public MDPSolver(Problem prob) {
      utilities = new double[prob.map.size()][prob.map.get(0).size()];
      rewards = new double[prob.map.size()][prob.map.get(0).size()];
      // Initialize utilities to the rewards in their spaces,
      // else 0
      for (int r = 0; r < utilities.length; r++) {
        for (int c = 0; c < utilities[0].length; c++) {
          String spaceContents = prob.map.get(r).get(c);
          if (spaceContents.equals("G")) {
            utilities[r][c] = GOLD_REWARD;
            rewards[r][c] = GOLD_REWARD;
          } else if (spaceContents.equals("P")) {
            utilities[r][c] = PIT_REWARD;
            rewards[r][c] = PIT_REWARD;
          } else {
            utilities[r][c] = 0.0;
            rewards[r][c] = 0.0;
          }
        }
      }
    }

    Policy solve(Problem prob, int iterations) {
      Policy policy = new Policy(prob);
      int rows = prob.map.size();
      int cols = prob.map.get(0).size();

      //iterate over the util map ITERATIONS times
      for (int i = 0; i < iterations; i++) {

        double[][] previous = new double[utilities.length][utilities.length];
        for (int u = 0; u < utilities.length; u++) {
          previous[u] = Arrays.copyOf(utilities[u], utilities[u].length);
        }

        for (int j = 0; j < rows; j++) {
          for (int k = 0; k < cols; k++) {
            ArrayList<ArrayList<Double>> cardinalValues = getCardinalValues(j, k, previous);
            IteratorOutput stateOutput = utilCalc(previous[j][k], rewards[j][k], prob.moveProbs, cardinalValues);
            utilities[j][k] = stateOutput.bestUtilValue;

            if (prob.map.get(j).get(k).equals("G")) {
              policy.bestActions[j][k] = "G";
            } else if (prob.map.get(j).get(k).equals("P")) {
              policy.bestActions[j][k] = "P";
            } else {
              policy.bestActions[j][k] = stateOutput.bestUtilDir;
            }
          }
        }
      }
      return policy;
    }

    private static ArrayList<ArrayList<Double>> getCardinalValues(int row, int col, double[][] utilMap) {
      ArrayList<ArrayList<Double>> output = new ArrayList<>();
      int numRows = utilMap.length;
      int numCols = utilMap[0].length;
      //find Up values
      ArrayList<Double> up = new ArrayList<>();
      for (int i = 1; i <= 3; i++) {
        if (row - i >= 0) {
          up.add(utilMap[row - i][col]);
        }
      }
      //find Right values
      ArrayList<Double> right = new ArrayList<>();
      for (int j = 1; j <= 3; j++) {
        if (col + j < numCols) {
          right.add(utilMap[row][col + j]);
        }
      }
      //find Down values
      ArrayList<Double> down = new ArrayList<>();
      for (int k = 1; k <= 3; k++) {
        if (row + k < numRows) {
          down.add(utilMap[row + k][col]);
        }
      }
      //find Left values
      ArrayList<Double> left = new ArrayList<>();
      for (int m = 1; m <= 3; m++) {
        if (col - m >= 0) {
          left.add(utilMap[row][col - m]);
        }
      }

      output.add(up);
      output.add(right);
      output.add(down);
      output.add(left);

      return output;
    }


    private static IteratorOutput utilCalc(double currentUtil, double currentReward, double[] moveProbs,
                                           ArrayList<ArrayList<Double>> cardinalValues) {
      IteratorOutput output = new IteratorOutput();

      double bestDirectionValue = -Double.MAX_VALUE;
      String bestDirectionString = "X";

      int moveProbSize = moveProbs.length;
      //try to find the best action: up, down, left, right
      for (int i = 0; i < directionStrings.length; i++) {
        ArrayList<Double> direction = cardinalValues.get(i);

        double directionValue = 0;
        //if a direction is a wall, util is the same as current state.
        if (direction.isEmpty()) {
          directionValue = currentUtil;
        }
        //otherwise, use Bellman's Eq.
        else {
        /*Find values in all directions for as long as we have probabilities. That is,
        if we have 3 probabilities, find the values of 3 next states in a given direction.
        If only 1 probability, find only the value of the next state in a given direction.
        */
          for (int j = 0; j < moveProbSize; j++) {
            //If we hit a wall before exhausting all probabilities, finish calculations
            //with last valid value in that direction.
            try {
              directionValue += moveProbs[j] * direction.get(j);
            } catch (IndexOutOfBoundsException E) {
              directionValue += moveProbs[j] * direction.get(direction.size() - 1);
            }
          }
        }
        //update best direction value as needed.
        if (directionValue > bestDirectionValue) {
          bestDirectionValue = directionValue;
          bestDirectionString = directionStrings[i];
        }
      }

      //calculate new utility of state.
      output.bestUtilValue = currentReward + DISCOUNT_FACTOR * bestDirectionValue;
      output.bestUtilDir = bestDirectionString;
      return output;
    }

    private static class IteratorOutput {
      double bestUtilValue;
      String bestUtilDir;
    }
  }


  // QLearner:  Same problem as MDP, but the agent doesn't know what the
  // world looks like, or what its actions do.  It can learn the utilities of
  // taking actions in particular states through experimentation, but it
  // has no way of realizing what the general action model is
  // (like "Right" increasing the column number in general).
  private static class QLearner {

    // Use these to index into the first index of utilities[][][]
    public static final int UP = 0;
    public static final int RIGHT = 1;
    public static final int DOWN = 2;
    public static final int LEFT = 3;
    public static final int ACTIONS = 4;

    public static int numRows;
    public static int numCols;

    public double utilities[][][];  // utilities of actions
    public double rewards[][];

    public QLearner(Problem prob) {
      numRows = prob.map.size();
      numCols = prob.map.get(0).size();

      utilities = new double[ACTIONS][numRows][numCols];
      // Rewards are for convenience of lookup; the learner doesn't
      // actually "know" they're there until encountering them
      rewards = new double[numRows][numCols];
      for (int r = 0; r < rewards.length; r++) {
        for (int c = 0; c < rewards[0].length; c++) {
          String locType = prob.map.get(r).get(c);
          if (locType.equals("G")) {
            rewards[r][c] = GOLD_REWARD;
          } else if (locType.equals("P")) {
            rewards[r][c] = PIT_REWARD;
          } else {
            rewards[r][c] = 0.0; // not strictly necessary to init
          }
        }
      }
      // Java: default init utilities to 0
    }

    public Policy solve(Problem prob, int iterations) {
      Policy policy = new Policy(prob);
      //row, col, action (up, right, left, down)

      for (int i = 0; i < iterations; i++) {
        int currentRow = rng.nextInt(numRows);
        int currentCol = rng.nextInt(numCols);
        do {

          //if Pit or Gold is found, update QMap and trigger ending condition for this iteration.
          if (rewards[currentRow][currentCol] == GOLD_REWARD) {
            utilities[UP][currentRow][currentCol] = GOLD_REWARD;
            utilities[RIGHT][currentRow][currentCol] = GOLD_REWARD;
            utilities[DOWN][currentRow][currentCol] = GOLD_REWARD;
            utilities[LEFT][currentRow][currentCol] = GOLD_REWARD;
            break;
          } else if (rewards[currentRow][currentCol] == PIT_REWARD) {
            utilities[UP][currentRow][currentCol] = PIT_REWARD;
            utilities[RIGHT][currentRow][currentCol] = PIT_REWARD;
            utilities[DOWN][currentRow][currentCol] = PIT_REWARD;
            utilities[LEFT][currentRow][currentCol] = PIT_REWARD;
            break;
          }
          //otherwise choose a direction either randomly or via bestQDirection.
          int direction = 0;
          if (rng.nextDouble() < EXPLORE_PROB) {
            direction = rng.nextInt(4);
          } else {
            direction = bestQDirection(utilities, currentRow, currentCol);
          }
          //choose how many tiles were slid based on the given probabilities.
          int tilesSlid = slideDistance(prob.moveProbs);

          int newRow = currentRow;
          int newCol = currentCol;
          //get the new state's row and column numbers.
          switch (direction) {
            case UP:
              if (newRow - tilesSlid >= 0) {
                newRow -= tilesSlid;
              } else {
                newRow = 0;
              }
              break;
            case RIGHT:
              if (newCol + tilesSlid < numCols) {
                newCol += tilesSlid;
              } else {
                newCol = numCols - 1;
              }
              break;
            case DOWN:
              if (newRow + tilesSlid < numRows) {
                newRow += tilesSlid;
              } else {
                newRow = numRows - 1;
              }
              break;
            case LEFT:
              if (newCol - tilesSlid > 0) {
                newCol -= tilesSlid;
              } else {
                newCol = 0;
              }
              break;
          }

          //find the best action for the new state.
          double bestQvalue = -Double.MAX_VALUE;
          for (int j = 0; j < 4; j++) {
            if (utilities[j][newRow][newCol] > bestQvalue) {
              bestQvalue = utilities[j][newRow][newCol];
            }
          }

          //Update the utility of the old state using the new one and the formula:
          // Q(a,s) = Q(a,s) + Learning Rate( Reward(s) +
          // Discount Factor(Max[Q(new state, all actions)]) - Q(a,s)
          utilities[direction][currentRow][currentCol] += LEARNING_RATE * (
                  rewards[currentRow][currentCol] + DISCOUNT_FACTOR * bestQvalue
                          - utilities[direction][currentRow][currentCol]);

          currentRow = newRow;
          currentCol = newCol;
        } while (true);
      }

      //trace through the finished Utilities Matrix and determine the best action in
      //each given state.
      for (int k = 0; k < numRows; k++) {
        for (int m = 0; m < numCols; m++) {
          double bestPolicyValue = -Double.MAX_VALUE;
          int bestPolicyDirection = 0;

          if (rewards[k][m] == GOLD_REWARD) {
            policy.bestActions[k][m] = "G";
          } else if (rewards[k][m] == PIT_REWARD) {
            policy.bestActions[k][m] = "P";
          } else {

            for (int n = 0; n < ACTIONS; n++) {
              if (utilities[n][k][m] > bestPolicyValue) {
                bestPolicyValue = utilities[n][k][m];
                bestPolicyDirection = n;
              }
            }
            switch (bestPolicyDirection) {
              case UP:
                policy.bestActions[k][m] = "U";
                break;
              case RIGHT:
                policy.bestActions[k][m] = "R";
                break;
              case DOWN:
                policy.bestActions[k][m] = "D";
                break;
              case LEFT:
                policy.bestActions[k][m] = "L";
                break;
            }
          }
        }
      }
      return policy;
    }

    //Returns 0 for up, 1 for right, 2 for down, 3 for left.
    private static int bestQDirection(double[][][] utilities, int row, int col) {
      double bestReward = -Double.MAX_VALUE;
      int bestDirection = 0;
      //0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
      for (int i = 0; i < 4; i++) {
        if (utilities[i][row][col] > bestReward) {
          bestReward = utilities[i][row][col];
          bestDirection = i;
        }
      }
      return bestDirection;
    }

    //returns the number of tiles the agent will slide: 1, 2, or 3.
    private static int slideDistance(double[] moveProbs) {
      if (moveProbs.length != 1) {

        int smallestProbDistance = -1;
        double smallestProb = -1;

        int midProbDistance = -1;
        double midProb = -1;

        int largestProbDistance = -1;
        double largestProb = -1;

        for (int i = 0; i < moveProbs.length; i++) {
          if (moveProbs[i] > largestProb) {
            largestProbDistance = i + 1;
            largestProb = moveProbs[i];
          } else if (moveProbs[i] > midProb && moveProbs[i] < largestProb) {
            midProbDistance = i + 1;
            midProb = moveProbs[i];
          } else {
            smallestProbDistance = i + 1;
            smallestProb = moveProbs[i];
          }
        }

        double slideValue = rng.nextDouble();

        if (slideValue <= smallestProb) {
          return smallestProbDistance;
        } else if (slideValue <= midProb + smallestProb) {
          return midProbDistance;
        } else {
          return largestProbDistance;
        }
      }

      return 1;
    }
  }
}

