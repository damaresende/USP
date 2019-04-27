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
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Test;


public class SudokuBoardBTFCTest {

	public static SudokuBoardBT answer;
	public static String boardFile; 
	
	/**
     * Setting up answer board and board file name
     */
	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		boardFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1.txt";
		
		String answerFile = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "board1_answer.txt";
		
		answer = new SudokuBoardBT();
		answer.fillData(answerFile);
	}
	
	/**
     * Tests if domain values are initialized to true for all values
     */
	@Test
	public void testBoardInitialization() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		for(int i = 0; i < board.getBoardSize(); i++) {
			for (int j = 0; j < board.getBoardSize(); j++) {
				assertNotNull(board.getDomain(i, j));
				assertEquals(0, board.getDomain(i, j).size());
			}
		}
	}
	
	/**
     * Tests if not valid values are removed from domain
     */
	@Test
	public void testFillDataDomain() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		board.fillData(boardFile);
		board.initDomain();
		
		String initialDomain = System.getProperty("user.dir") + File.separator + "test" + File.separator 
				+ "boards" + File.separator + "initial_domain.txt";
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(initialDomain));
			String line = "";
		
			int i = 0;
			while ((line = reader.readLine()) != null) {
				String [] parts = line.trim().split(" ");
			
				for (int k = 0; k < parts.length; k++) {
					LinkedList<Integer> domain = board.getDomain(i);
					assertTrue(domain.get(k) == Integer.parseInt(parts[k]));
				}
				i++;
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("ERROR: Domain file could not be read");
		}
	}

	/**
     * Tests if update domain is updating the row of the cell a value is set
     */
	@Test
	public void testUpdateDomainCellRow() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		board.fillData(boardFile);
		board.initDomain();
		board.updateDomain(5, 2, 7);
		
		assertEquals(1, board.getDomain(5, 2).size());
		assertTrue(7 == board.getDomain(5, 2).getFirst());
	}
	
	/**
     * Tests if update domain is updating the domain of cells in the
     * same column of the cell that was set
     */
	@Test
	public void testUpdateDomainCleanRows() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		board.fillData(boardFile);
		board.initDomain();	
		board.updateDomain(2, 2, 4);
		
		// row to set to one value
		assertEquals(1, board.getDomain(2, 2).size());
		assertTrue(4 == board.getDomain(2, 2).getFirst());
		
		// rows to keep
		assertEquals(2, board.getDomain(2).size());
		assertTrue(3 == board.getDomain(2).get(0));
		assertTrue(9 == board.getDomain(2).get(1));
		
		assertEquals(1, board.getDomain(11).size());
		assertTrue(8 == board.getDomain(11).get(0));
		
		// rows to clean
		int [] rowsToClean = new int[]{29, 38, 47, 54, 65, 74};
		for (int row : rowsToClean) {
			assertEquals(-1, board.getDomain(row).indexOf(4));
		}
		
	}
	
	/**
     * Tests if update domain is updating the domain of cells in the
     * same row of the cell that was set
     */
	@Test
	public void testUpdateDomainCleanColumns() {
		SudokuBoardBTFC board = new SudokuBoardBTFC();
		
		board.fillData(boardFile);
		board.initDomain();		
		board.updateDomain(2, 2, 4);
		
		// row to set to one value
		assertEquals(1, board.getDomain(2, 2).size());
		assertTrue(4 == board.getDomain(2, 2).getFirst());
		
		// columns to keep
		assertEquals(1, board.getDomain(18).size());
		assertTrue(2 == board.getDomain(18).get(0));
		
		assertEquals(4, board.getDomain(19).size());
		assertTrue(3 == board.getDomain(19).get(0));
		assertTrue(5 == board.getDomain(19).get(1));
		assertTrue(7 == board.getDomain(19).get(2));
		assertTrue(9 == board.getDomain(19).get(3));
		
		// columns to clean
		int [] columnsToClean = new int[]{21, 22, 23, 24, 25, 26};
		for (int col : columnsToClean) {
			assertEquals(-1, board.getDomain(col).indexOf(4));
		}
		
	}
}
