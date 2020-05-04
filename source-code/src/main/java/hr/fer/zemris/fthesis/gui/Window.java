package hr.fer.zemris.fthesis.gui;

import hr.fer.zemris.fthesis.ann.dataset.model.ClassType;
import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

import static hr.fer.zemris.fthesis.ann.dataset.model.ClassType.*;

public class Window extends JFrame {

    // ----Canvas component---- //
    private static final int CANVAS_WIDTH = 500;
    private static final int CANVAS_HEIGHT = 500;
    private final List<Sample> samples = new ArrayList<>();
    private JComponent myCanvas;
    // ------------------------ //

    // ----OPTIONS PANEL---- //
    private JPanel panelOptions;
    private JLabel lblHiddenLayers;
    private JTextField fldHiddenLayers;
    private JLabel lblClassType;
    private JTextField fldClassType;
    private JLabel lblEta;
    private JTextField fldEta;
    private JLabel lblIter;
    private JTextField fldIter;
    private JLabel lblErr;
    private JTextField fldErr;
    private JRadioButton btnSigmoid;
    private JRadioButton btnReLu;
    private JRadioButton btnTanh;
    private JButton btnTrain;
    private JButton btnStop;
    // -------------------- //

    // --- Class type stuff --- //
    private final List<ClassType> classTypes = List.of(CLASS_A, CLASS_B, CLASS_C);
    private ClassType currentClassType = CLASS_A;
    // ------------------------ //

    public Window() {
        setTitle("ANN classification");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        initGUI();
        pack();
        setLocationRelativeTo(null);
    }

    private void initGUI() {
        initPanelOptions();
        initCanvasComponent();
    }

    // ---- INITIALIZE OPTIONS PANEL ! ----- //
    private void initPanelOptions() {
        panelOptions = new JPanel(new GridLayout(5, 1));
        addFirstRow();
        addSecondRow();
        addThirdRow();
        addFourthRow();
        addFifthRow();
        getContentPane().add(panelOptions, BorderLayout.SOUTH);
    }

    private void addFirstRow() {
        JPanel panelHiddenLayers = new JPanel();
        lblHiddenLayers = new JLabel("Hidden layers:");
        fldHiddenLayers = new JTextField("3,3,3", 10);
        panelHiddenLayers.add(lblHiddenLayers);
        panelHiddenLayers.add(fldHiddenLayers);
        panelOptions.add(panelHiddenLayers);
    }

    private void addSecondRow() {
        JPanel panelParams = new JPanel();
        lblIter = new JLabel("IterLimit:");
        fldIter = new JTextField("100000", 7);
        lblEta = new JLabel("Eta:");
        fldEta = new JTextField("0,1", 5);
        lblErr = new JLabel("MaxError:");
        fldErr = new JTextField("0.02", 5);
        panelParams.add(lblIter);
        panelParams.add(fldIter);
        panelParams.add(lblEta);
        panelParams.add(fldEta);
        panelParams.add(lblErr);
        panelParams.add(fldErr);
        panelOptions.add(panelParams);
    }

    private void addThirdRow() {
        JPanel panelClassType = new JPanel();
        lblClassType = new JLabel("Class type:");
        fldClassType = new JTextField(6);
        fldClassType.setEditable(false);
        fldClassType.getDocument().addDocumentListener(new DocumentListener() {
            public void insertUpdate(DocumentEvent e) {
                fldClassType.setBackground(currentClassType.getColor());
            }

            public void removeUpdate(DocumentEvent e) {
            }

            public void changedUpdate(DocumentEvent e) {
            }
        });
        fldClassType.setText(currentClassType.name());
        panelClassType.add(lblClassType);
        panelClassType.add(fldClassType);
        panelOptions.add(panelClassType);
    }

    private void addFourthRow() {
        JPanel panelButtons = new JPanel();
        ButtonGroup group = new ButtonGroup();
        btnSigmoid = new JRadioButton("Sigmoid neuron");
        btnSigmoid.setSelected(true);
        btnReLu = new JRadioButton("ReLu neuron");
        btnTanh = new JRadioButton("Tanh neuron");
        group.add(btnSigmoid);
        group.add(btnReLu);
        group.add(btnTanh);
        panelButtons.add(btnSigmoid);
        panelButtons.add(btnReLu);
        panelButtons.add(btnTanh);
        panelOptions.add(panelButtons);
    }

    private void addFifthRow() {
        JPanel panelButtons = new JPanel();
        btnTrain = new JButton("Train");
        btnTrain.setFocusPainted(false);
        btnStop = new JButton("Stop");
        btnStop.setFocusPainted(false);
        btnStop.setEnabled(false);
        panelButtons.add(btnTrain);
        panelButtons.add(btnStop);
        panelOptions.add(panelButtons);
    }
    // ------------------------------------- //

    private void initCanvasComponent() {
        myCanvas = new CanvasComponent(CANVAS_WIDTH, CANVAS_HEIGHT, samples);
        myCanvas.addMouseListener(new MouseAdapter() {
            int classTypesIndex = 0;

            public void mouseClicked(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    if (classTypesIndex == classTypes.size() - 1) {
                        classTypesIndex = 0;
                    } else {
                        classTypesIndex++;
                    }
                    currentClassType = classTypes.get(classTypesIndex);
                    fldClassType.setText(currentClassType.name());
                } else {
                    double[] inputs = {e.getX(), e.getY()};
                    double[] outputs = currentClassType.getOutputs();
                    Sample sample = new Sample(inputs, outputs);
                    sample.setClassType(currentClassType);
                    samples.add(sample);
                    myCanvas.repaint();
                }
            }
        });
        getContentPane().add(myCanvas);
    }

    private static class CanvasComponent extends JComponent {

        private final List<Sample> samples;

        public CanvasComponent(int width, int height, List<Sample> samples) {
            this.samples = samples;
            setPreferredSize(new Dimension(width, height));
            setBorder(new TitledBorder(new EtchedBorder(), "Canvas"));
        }

        @Override
        protected void paintComponent(Graphics g) {
            Rectangle2D grid = rectangularGrid();
            g = g.create(grid.x, grid.y, grid.width, grid.height);

            // clear background
            g.setColor(Color.WHITE);
            g.fillRect(0, 0, grid.width, grid.height);

            g.setColor(Color.BLACK);
            for (Sample s : samples) {
                double[] inputs = s.getInputs();
                int x = (int) inputs[0] - grid.x;
                int y = (int) inputs[1] - grid.y;
                g.drawRect(x, y, 5, 5);

            }

        }

        private Rectangle2D rectangularGrid() {
            Insets insets = getInsets();
            Dimension dimension = getSize();
            int x = insets.left;
            int y = insets.top;
            int width = dimension.width - insets.left - insets.right - 1;
            int height = dimension.height - insets.top - insets.bottom;
            return new Rectangle2D(x, y, width, height);
        }

    }

}
