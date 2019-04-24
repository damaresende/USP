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
	
	private int[][] board;
	private int boardDim = 3;
	private int boardSize = boardDim * boardDim;
	private Queue<Coordinates> toComplete = new LinkedList<Coordinates>();
	
	/**
     * Creates a 9x9 matrix
     */
	public SudokuBoard() {
		board = new int[boardSize][boardSize];
	}
	
	/**
     * Reads data from the specified file and fills each cell of the Sudoku board
     * with the values indicated. Only 9x9 boards are accepted.
     * 
     * @param boardFile: string with board file path and name
     * @return true if board was successfully filled, false otherwise
     */
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
	
	/**
     * Fills the board empty values with a valid value
     * 
     * @param cell: first cell to fill
     * @return true if there are no values to be filled, false otherwise
     */
	public boolean backtracking(Coordinates cell) {
		if (toComplete.size() == 0) {
			for(int i = 1; i <= boardSize; i++) {
				if (evaluate(cell, i)) {
					board[cell.getI()][cell.getJ()] = i;
				}
			}
			return true;
		}
		
		for(int i = 1; i <= boardSize; i++) {
			if (evaluate(cell, i)) {
				board[cell.getI()][cell.getJ()] = i;
				
				if (backtracking(toComplete.poll())) {
					return true;
				} else {
					board[cell.getI()][cell.getJ()] = -1;
				}
					
			}
		}
		toComplete.add(cell);
		return false;
	}
	
	/**
     * Checks if the specified cell can hold the specified value according to the
     * rules of Sudoku game.
     * 
     * @param cell: cell to fill
     * @param value: value to place in cell
     * @return true if value is valid, false otherwise
     */
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
	
	/**
     * Checks if the list of cells to be completed is empty.
     * 
     * @return true if list is empty, false otherwise
     */
	public boolean toCompleteIsEmpty() {
		if (toComplete.size() == 0)
			return true;
		return false;
	}
	
	/**
     * Gets cell value based on the specified coordinates
     * 
     * @param i: row coordinate
     * @param j: column coordinate
     * @return cell value
     */
	public int getCellValue(int i, int j) {
		return board[i][j];
	}

	/**
     * Retrieves board size. It is always 9.
     * 
     * @return board size
     */
	public int getBoardSize() {
		return boardSize;
	}

	/**
     * Pulls from the queue of cells to be completed the next cell
     * that need to be filled up.
     * 
     * @return cell that needs to be filled up
     */
	public Coordinates getNextCellToFill() {
		return toComplete.poll();
	}
	
}
