/*
 * Solves 9x9 Sudoku boards by making use of backtracking algorithm.
 * 3 parameters can be passed as input argument: bt, fc and mvr.
 * Only bt method is working.
 * 
 * @author: Damares Resende
 * @contact: damaresresende@usp.br
 * @since: Apr 28, 2019
 * 
 * @organization: University of Sao Paulo (USP)
 *     Institute of Mathematics and Computer Science (ICMC)
 *     Project of Algorithms Class (SCC5000)
*/
package sudoku;

import java.util.Scanner;

public class Main {

	public static void main(String[] args) {
		if (args.length < 1) {
			System.out.println("ERROR. You must provide at least one option to solve Sudoku. "
					+ "Please try 'bt', 'fc' or 'mvr'.");
			return;
		}
		
		SudokuBoardBT board; 
		if (args[0].equals("bt")) {
			System.out.println("Solving Sudoku by using backtracking algorithm.");
			board = new SudokuBoardBT(3);
			
		} else if (args[0].equals("fc")) {
			System.out.println("Solving Sudoku by using backtracking with forward checking prunning.");
			System.out.println("Solution did not work. :'(");
			board = new SudokuBoardBTFC(3);
			return;
			
		} else if (args[0].equals("mvr")) {
			System.out.println("Solving Sudoku by using backtracking with forward checking and MVR prunning.");
			System.out.println("Solution was not implemented. =/");
			return;
		} else {
			System.out.println("ERROR. Not a valid option. Please try 'bt', 'fc' or 'mvr'.");
			return;
		}
		
		Scanner in = new Scanner(System.in);
		
		int numOfTests = in.nextInt();
		in.skip("(\r\n|[\n\r\u2028\u2029\u0085])?");
		
		System.out.println("\nA total of " + numOfTests + " boards will be tested.");
		
		for (int k = 1; k <= numOfTests; k++) {
			System.out.println("\nSolving board number " + k + "...\n");
			
			String line = in.nextLine().trim();
			while (!line.equals("") && in.hasNextLine()) {
				System.out.println(line);
				board.fillData(line);
				line = in.nextLine().trim();
			}
			
			if(board.resolve()) {
				System.out.println("\nA solution was found after " + board.getNumOfAttributions() + " attributions.\n");
				board.printBoard();
			} else {
				if (board.isSaturated())
					System.out.println("\nNumber of attributions exceeds the maximum limit.\n");
				else
					System.out.println("\nBoard has no solution. " + board.getNumOfAttributions() + " attributions where made.\n");
			}
			board.reset(3);
			System.gc();
		}
		in.close();
	}

}
