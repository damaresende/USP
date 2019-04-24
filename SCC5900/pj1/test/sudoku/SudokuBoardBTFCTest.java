/*
 * Unit tests for creating and filling up a Sudoku board
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

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.junit.BeforeClass;
import org.junit.Test;


public class SudokuBoardBTFCTest {

	public static SudokuBoardBT answer;
	public static String boardFile; 
	public static boolean [][] domain;
	
	/**
     * Setting up answer board and board file name
     */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		boardFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1.txt";
		
		String answerFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1_answer.txt";
		
		String initialDomain = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "initial_domain.txt";
		
		answer = new SudokuBoardBT();
		answer.fillData(answerFile);
		
		domain = new boolean[answer.getBoardSize() * answer.getBoardSize()][answer.getBoardSize()];
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(initialDomain));
			String line = "";
			int i = 0;
			
			while ((line = reader.readLine()) != null) {
				String [] parts = line.trim().split(" ");
				
				for (int j = 0; j < answer.getBoardSize(); j++) {
					if (parts[j].equals("1"))
						domain[i][j] = true;
				}
				i++;
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("ERROR: Domain file could not be read");
		}
	}
	
	/**
     * Tests if domain values are initialized to true for all values
     */
	@Test
	public void testBoardInitialization() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		for(int i = 0; i < board.getBoardSize(); i++) {
			for (int j = 0; j < board.getBoardSize(); j++) {
				boolean [] domain = board.getDomain(i, j);
				
				for (int k = 0; k < board.getBoardSize(); k++) {
					assertTrue(domain[k]);
				}
			}
		}
	}
	
	/**
     * Tests if not valid values are removed from domain in fill data method
     */
	@Test
	public void testFillDataDomain() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		board.fillData(boardFile);
		
		for(int i = 0; i < board.getBoardSize() * board.getBoardSize(); i++) {
			for (int j = 0; j < board.getBoardSize(); j++) {
				try {
					assertEquals(domain[i][j], board.domain[i][j]);
				} catch (AssertionError e) {
					System.out.println("Error: [" + i + "," + j + "]. " + domain[i][j] + " != " + board.domain[i][j]);
				}
			}
		}
	}
}
