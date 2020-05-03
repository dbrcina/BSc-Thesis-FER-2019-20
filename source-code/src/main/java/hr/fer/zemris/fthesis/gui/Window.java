package hr.fer.zemris.fthesis.gui;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;

public class Window extends JFrame {

    private static final int CANVAS_WIDTH = 500;
    private static final int CANVAS_HEIGHT = 500;

    private final JComponent myCanvas = new CanvasPanel(CANVAS_WIDTH, CANVAS_HEIGHT);
    private final JPanel options = new JPanel();

    public Window() {
        setTitle("ANN classification");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        initGUI();
        pack();
        setLocationRelativeTo(null);
    }

    private void initGUI() {
        Container contentPane = getContentPane();
        contentPane.add(myCanvas);
        contentPane.add(options, BorderLayout.SOUTH);
    }

    private static class CanvasPanel extends JComponent {
        public CanvasPanel(int width, int height) {
            setPreferredSize(new Dimension(width, height));
            setBorder(new TitledBorder(new EtchedBorder(), "Canvas"));
        }

        @Override
        protected void paintComponent(Graphics g) {
            Rectangle2D rectangle = calculateRectangle();

            // clear background
            g.setColor(Color.WHITE);
            g.fillRect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
        }

        private Rectangle2D calculateRectangle() {
            Insets insets = getInsets();
            Dimension dimension = getSize();
            int x = insets.left;
            int y = insets.top;
            int width = dimension.width - insets.left - insets.right;
            int height = dimension.height - insets.top - insets.bottom;
            return new Rectangle2D(x, y, width, height);
        }

    }

}
