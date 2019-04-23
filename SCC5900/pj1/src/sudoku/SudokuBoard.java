/*
 * Creates a Sudoku board based on a matrix in a text file.
 * Only 9 x 9 boards are accepted.
 * 
 * @author: Damares Resende
 * @contact: damaresresende@usp.br
 * @since: Apr 14, 2019
 * 
 * @organization: University of Sao Paulo (USP)
 *     Institute of Mathematics and Computer Science (ICMC)
 *     Project of Algorithms Class (SCC5000)
*/

package sudoku;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;

public class SudokuBoard {
	
	int[][] board;
	int boardDim = 3;
	int boardSize = boardDim * boardDim;
	Queue<Coordinates> toComplete = new LinkedList<Coordinates>();
	
	public SudokuBoard() {
		board = new int[boardSize][boardSize];
	}
	
	public boolean fillData(String boardFile) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(boardFile));
			String line = "";
			int i = 0;
			
			while ((line = reader.readLine()) != null) {
				String [] parts = line.trim().split(" ");
				
				if (parts.length != boardSize) {
					System.out.println("ERROR: The number of board values per row should be "  
							+ String.valueOf(boardSize) + ".");
					reader.close();
					return false;
				}
				
				for (int j = 0; j < boardSize; j++) {
					if (parts[j].equals("-")) {
						board[i][j] = -1;
						toComplete.add(new Coordinates(i, j));
					}
					else {
						board[i][j] = Integer.parseInt(parts[j]);
						
						if (board[i][j] < 0) {
							System.out.println("ERROR: board values must be greater than 0.");
							reader.close();
							return false;
						}
					}
				}
				i++;
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("ERROR: Board file could not be read");
			return false;
		} catch (NumberFormatException e) {
			System.out.println("ERROR: Board value could not be parsed");
			return false;
		}
		return true;
	}
	
	public boolean backtracking(Coordinates cell) {
		if (toComplete.poll() == null)
			return true;
		
		for(int i = 0; i < boardSize; i++) {
			if (evaluate(cell, i)) {
				board[cell.getI()][cell.getJ()] = i;
				return backtracking(toComplete.poll());
			}
		}
		toComplete.add(cell);
		return false;
	}
	
	public boolean evaluate(Coordinates cell, int value) {
		for (int j = 0; j < boardSize; j++) {
			if (board[cell.getI()][j] == value)
				return false;
		}
		
		for (int i = 0; i < boardSize; i++) {
			if (board[i][cell.getJ()] == value)
				return false;
		}
		
		for (int i = cell.getPivotI(); i < cell.getPivotI() + boardDim; i++) {
			for (int j = cell.getPivotJ(); j < cell.getPivotJ() + boardDim; j++) {
				if (board[i][j] == value)
					return false;
			}
		}
		
	
		return true;
	}
	
}
