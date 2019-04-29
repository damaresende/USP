/*
 * Creates a Sudoku board based on a matrix in a text file and uses
 * backtracking to solve it.
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

import java.util.LinkedList;
import java.util.Queue;


public class SudokuBoardBT {
	
	private int row;
	protected int[][] board;
	protected int boardDim;
	protected int boardSize;
	private boolean isSaturated;
	private int numOfAttributions;

	protected Queue<Coordinates> toComplete;
	
	/**
     * Creates a dim x dim matrix
     * 
     * @param dim: dimension of the Sudoku board
     */
	public SudokuBoardBT(int dim) {
		this.row = 0;
		this.boardDim = dim;
		this.isSaturated = false;
		this.numOfAttributions = 0;
		this.boardSize = boardDim * boardDim;
		
		board = new int[boardSize][boardSize];
		toComplete = new LinkedList<Coordinates>();
	}
	
	/**
     * Resets board structure 
     */
	public void reset(int dim) {
		this.row = 0;
		this.boardDim = dim;
		this.isSaturated = false;
		this.numOfAttributions = 0;
		this.boardSize = boardDim * boardDim;
		
		board = new int[boardSize][boardSize];
		toComplete = new LinkedList<Coordinates>();
	}
	
	/**
     * Reads data from the specified file and fills each cell of the Sudoku board
     * with the values indicated.
     * 
     * @param line: sudoku input line
     * @return true if board was successfully filled, false otherwise
     */
	public boolean fillData(String line) {
		try {
			String [] parts = line.trim().split(" ");
			
			if (parts.length != boardSize) {
				System.out.println("ERROR: The number of board values per row should be " + boardSize + ".");
				return false;
			}
			
			for (int j = 0; j < boardSize; j++) {
				board[row][j] = Integer.parseInt(parts[j]);
				
				if (board[row][j] > boardSize || board[row][j] < 0) {
					System.out.println("ERROR: Board values must be in between 0 and "  + boardSize + ".");
					return false;
				}
				
				if (board[row][j] == 0) {
					toComplete.add(new Coordinates(row, j));
				}
			}
			row++;
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
		for(int i = 1; i <= boardSize; i++) {
			if (evaluate(cell, i)) {
				board[cell.getI()][cell.getJ()] = i;
				this.numOfAttributions++;
				
				if (numOfAttributions > 10e6) {
					this.isSaturated = true;
					return false;
				}
				
				if (toComplete.size() == 0) {
					return true;
				}
				
				if (backtracking(toComplete.poll())) {
					return true;
				} else {
					board[cell.getI()][cell.getJ()] = 0;
				}
			}
		}
		
		toComplete.add(cell);
		return false;
	}
	
	/**
     * Solves Sudoku by using backtracking
     */
	public boolean resolve() {
		return backtracking(getNextCellToFill());
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
     * Retrieves the number of attributions
     * 
     * @return number of attributions
     */
	public int getNumOfAttributions() {
		return numOfAttributions;
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
	
	/**
     * Saturation flag
     * 
     * @return true if number of attributions exceeds 10e6
     */
	public boolean isSaturated() {
		return isSaturated;
	}
	
	/**
     * Prints Sudoku board
     */
	public void printBoard() {
		for(int i = 0; i < boardSize; i++) {
			for(int j = 0; j < boardSize; j++) {
				System.out.print(board[i][j] + " ");
			}
			System.out.println();
		}
	}
}
