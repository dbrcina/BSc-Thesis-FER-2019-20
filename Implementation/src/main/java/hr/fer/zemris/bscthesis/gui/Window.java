package hr.fer.zemris.bscthesis.gui;

import hr.fer.zemris.bscthesis.ann.NeuralNetwork;
import hr.fer.zemris.bscthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.bscthesis.classes.ClassType;
import hr.fer.zemris.bscthesis.dataset.Cartesian2DDataset;
import hr.fer.zemris.bscthesis.dataset.Dataset;
import hr.fer.zemris.bscthesis.dataset.Sample;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.List;
import java.util.*;

/**
 * Window used for results visualisation.
 *
 * @author dbrcina
 */
public class Window extends JFrame {

    /* ------------- NEURAL NETWORK STUFF ------------- */
    private final NeuralNetwork nn = new NeuralNetwork();
    private final Map<String, ActivationFunction> aFunctions = ActivationFunction.allAFunctions();
    private final Dataset dataset = new Cartesian2DDataset();
    private final List<ClassType> classTypes = new ArrayList<>(ClassType.allClassTypes());
    private final Map<String, NeuralNetwork.LearningType> learningTypes = new HashMap<>() {{
        put("Stochastic", NeuralNetwork.LearningType.ONLINE);
        put("Batch", NeuralNetwork.LearningType.BATCH);
        put("Mini-Batch", NeuralNetwork.LearningType.MINI_BATCH);
    }};
    private final List<Sample> samples = new ArrayList<>();
    private final List<Sample> normalizedSamples = new ArrayList<>();
    private ClassType currentClassType = classTypes.get(0);
    private NeuralNetwork.LearningType learningType;
    private int batchSize;
    /* ----------------------------------------------- */

    /* -------------- CANVAS COMPONENT --------------- */
    private static final int CANVAS_WIDTH = 500;
    private static final int CANVAS_HEIGHT = 400;
    private JComponent canvas;
    /* ----------------------------------------------- */

    /* ---------------- OPTIONS PANEL ---------------- */
    private JPanel panelOptions;
    private JLabel lblBPType;
    private JComboBox<String> boxBPType;
    private JLabel lblHiddenLayers;
    private JTextField fldHiddenLayers;
    private JLabel lblEta;
    private JTextField fldEta;
    private JLabel lblIter;
    private JTextField fldIter;
    private JLabel lblErr;
    private JTextField fldErr;
    private JLabel lblClassType;
    private JTextField fldClassType;
    private ButtonGroup btnGroup;
    private JRadioButton[] btnAFunctions;
    private JButton btnTrain;
    private JButton btnStop;
    private JButton btnDeleteLast;
    private JButton btnClearAll;
    /* ----------------------------------------------- */

    /* --------------- HELPER VARIABLES -------------- */
    private boolean training;
    /* ----------------------------------------------- */

    /* ----------------- CONSTRUCTOR ----------------- */
    public Window() {
        setResizable(false);
        setTitle("ANN classification");
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        initGUI();
        pack();
        setLocationRelativeTo(null);
    }
    /* ----------------------------------------------- */

    /* ---------- INITIALIZE GUI COMPONENTS ---------- */
    private void initGUI() {
        initPanelOptions();
        initCanvasComponent();
    }

    private void initPanelOptions() {
        panelOptions = new JPanel(new GridLayout(6, 1));
        addFirstRow();
        addSecondRow();
        addThirdRow();
        addFourthRow();
        addFifthRow();
        addSixthRow();
        getContentPane().add(panelOptions, BorderLayout.SOUTH);
    }

    private void addFirstRow() {
        JPanel panelComboBox = new JPanel();
        lblBPType = new JLabel("Backpropagation type:");
        boxBPType = new JComboBox<>(learningTypes.keySet().toArray(new String[0]));
        panelComboBox.add(lblBPType);
        panelComboBox.add(boxBPType);
        panelOptions.add(panelComboBox);
    }

    private void addSecondRow() {
        JPanel panelHiddenLayers = new JPanel();
        lblHiddenLayers = new JLabel("Hidden layers:");
        fldHiddenLayers = new JTextField("5,5", 10);
        panelHiddenLayers.add(lblHiddenLayers);
        panelHiddenLayers.add(fldHiddenLayers);
        panelOptions.add(panelHiddenLayers);
    }

    private void addThirdRow() {
        JPanel panelParams = new JPanel();
        lblIter = new JLabel("Epochs:");
        fldIter = new JTextField("100000", 7);
        lblEta = new JLabel("Eta:");
        fldEta = new JTextField("0.1", 5);
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

    private void addFourthRow() {
        JPanel panelClassType = new JPanel();
        lblClassType = new JLabel("Class type:");
        fldClassType = new JTextField(6);
        fldClassType.setEditable(false);
        fldClassType.setForeground(Color.WHITE);
        fldClassType.getDocument().addDocumentListener(new DocumentListener() {
            public void insertUpdate(DocumentEvent e) {
                fldClassType.setBackground(currentClassType.getColor());
            }

            public void removeUpdate(DocumentEvent e) {
            }

            public void changedUpdate(DocumentEvent e) {
            }
        });
        fldClassType.setText(currentClassType.getId());
        panelClassType.add(lblClassType);
        panelClassType.add(fldClassType);
        panelOptions.add(panelClassType);
    }

    private void addFifthRow() {
        JPanel panelButtons = new JPanel();
        btnGroup = new ButtonGroup();
        btnAFunctions = new JRadioButton[aFunctions.size()];
        int i = 0;
        for (String aFunctionId : aFunctions.keySet()) {
            JRadioButton btnAFunction = new JRadioButton(aFunctionId + " neuron");
            btnGroup.add(btnAFunction);
            panelButtons.add(btnAFunction);
            btnAFunctions[i++] = btnAFunction;
        }
        btnAFunctions[0].setSelected(true);
        panelOptions.add(panelButtons);
    }

    private void addSixthRow() {
        JPanel panelButtons = new JPanel();
        btnTrain = new JButton("Train");
        btnTrain.setFocusPainted(false);
        btnStop = new JButton("Stop");
        btnStop.setFocusPainted(false);
        btnStop.setEnabled(false);
        btnDeleteLast = new JButton("Delete last sample");
        btnDeleteLast.setFocusPainted(false);
        btnDeleteLast.setEnabled(false);
        btnClearAll = new JButton("Clear all samples");
        btnClearAll.setFocusPainted(false);
        btnClearAll.setEnabled(false);
        panelButtons.add(btnTrain);
        panelButtons.add(btnStop);
        panelButtons.add(btnDeleteLast);
        panelButtons.add(btnClearAll);
        panelOptions.add(panelButtons);
        addActionsToButtons();
    }

    private void addActionsToButtons() {
        btnClearAll.addActionListener(evt -> {
            samples.clear();
            btnDeleteLast.setEnabled(false);
            btnClearAll.setEnabled(false);
            training = false;
            canvas.repaint();
        });
        btnDeleteLast.addActionListener(evt -> {
            if (samples.size() == 1) {
                btnClearAll.getActionListeners()[0].actionPerformed(evt);
            } else {
                samples.remove(samples.size() - 1);
                canvas.repaint();
            }
        });
        btnTrain.addActionListener(evt -> {
            parseAndSetLayers();
            parseAndSetAFunction();
            parseAndSetDatasetSamples();
            parseAndSetLearningType();
            parseAndSetBatchSize();
            parseAndSetVisualisationParams();
            parseAndStartTraining();
        });
        btnStop.addActionListener(evt -> nn.stop());
    }

    private void parseAndSetLayers() {
        int[] hiddenLayers = Arrays.stream(fldHiddenLayers.getText().split(","))
                .mapToInt(Integer::parseInt)
                .filter(i -> i > 0)
                .toArray();
        int[] layers = new int[1 + hiddenLayers.length + 1];
        // input layer
        layers[0] = 2;
        // output layer
        layers[layers.length - 1] = classTypes.size();
        // hidden layers
        int offset = 1;
        for (int hLayer : hiddenLayers) {
            layers[offset++] = hLayer;
        }
        nn.setLayers(layers);
    }

    private void parseAndSetAFunction() {
        Enumeration<AbstractButton> buttons = btnGroup.getElements();
        while (buttons.hasMoreElements()) {
            AbstractButton btn = buttons.nextElement();
            if (btn.isSelected()) {
                // ID neuron
                String aFunctionId = btn.getText().split("\\s+")[0];
                nn.setAFunction(aFunctions.get(aFunctionId));
                break;
            }
        }
    }

    private void parseAndSetDatasetSamples() {
        normalizedSamples.clear();
        for (Sample s : samples) {
            double[] sInputs = s.getInputs();
            double[] inputs = {transformX(sInputs[0]), transformY(sInputs[1])};
            normalizedSamples.add(new Sample(inputs, s.getOutputs(), s.getClassType()));
        }
        dataset.setSamples(normalizedSamples);
        nn.setDataset(dataset);
    }

    private void parseAndSetLearningType() {
        String type = (String) boxBPType.getSelectedItem();
        learningType = learningTypes.get(type);
        nn.setLearningType(learningType);
    }

    private void parseAndSetBatchSize() {
        if (learningType == NeuralNetwork.LearningType.MINI_BATCH) {
            do {
                String bSize = JOptionPane.showInputDialog(this, "Enter batch size:");
                if (bSize == null) {
                    break;
                }
                if (!bSize.matches("^[1-9]\\d*$")) {
                    JOptionPane.showMessageDialog(this, "Enter natural number.");
                } else {
                    batchSize = Integer.parseInt(bSize);
                    break;
                }
            } while (true);
        }
        nn.setBatchSize(batchSize);
    }

    private void parseAndSetVisualisationParams() {
        nn.setCanvas(canvas);
        nn.setRedrawEveryNEpoch(1000);
    }

    private void parseAndStartTraining() {
        int epochs = Integer.parseInt(fldIter.getText());
        double maxError = Double.parseDouble(fldErr.getText());
        double eta = Double.parseDouble(fldEta.getText());
        new Thread(() -> {
            training = true;
            nn.train(epochs, maxError, eta);
            btnStop.setEnabled(false);
            canvas.repaint();
        }).start();
        btnStop.setEnabled(true);
    }
    /* ----------------------------------------------- */

    /* --------- INITIALIZE CANVAS COMPONENT --------- */
    private void initCanvasComponent() {
        canvas = new CanvasComponent(this, CANVAS_WIDTH, CANVAS_HEIGHT);
        canvas.addMouseListener(new MouseAdapter() {
            private int classTypesIndex = 0;

            public void mouseClicked(MouseEvent e) {
                if (SwingUtilities.isRightMouseButton(e)) {
                    if (classTypesIndex == classTypes.size() - 1) {
                        classTypesIndex = 0;
                    } else {
                        classTypesIndex++;
                    }
                    currentClassType = classTypes.get(classTypesIndex);
                    fldClassType.setText(currentClassType.getId());
                } else {
                    double[] inputs = {e.getX(), e.getY()};
                    double[] outputs = currentClassType.getDesiredOutputs();
                    Sample sample = new Sample(inputs, outputs, currentClassType);
                    samples.add(sample);
                    if (!btnDeleteLast.isEnabled()) {
                        btnDeleteLast.setEnabled(true);
                    }
                    if (!btnClearAll.isEnabled()) {
                        btnClearAll.setEnabled(true);
                    }
                    canvas.repaint();
                }
            }
        });
        getContentPane().add(canvas);
    }
    /* ----------------------------------------------- */

    /* --------------- HELPER METHODS ---------------- */
    private double transformX(double x) {
        return x / canvas.getSize().getWidth();
    }

    private double transformY(double y) {
        return y / canvas.getSize().getHeight();
    }
    /* ----------------------------------------------- */

    /**
     * Canvas which represents Cartesian 2D coordinate system.
     */
    private static class CanvasComponent extends JComponent {

        private final Window window;

        public CanvasComponent(Window window, int width, int height) {
            this.window = window;
            setPreferredSize(new Dimension(width, height));
            setBorder(new TitledBorder(new EtchedBorder(), "Sample space"));
        }

        @Override
        protected void paintComponent(Graphics g) {
            Rectangle grid = rectangularGrid();
            Graphics2D g2d = (Graphics2D) g.create(grid.x, grid.y, grid.width, grid.height);

            // clear background
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, grid.width, grid.height);

            if (window.training) {
                drawTraining(g2d, grid);
            }
            drawControlPoints(g2d, grid, window.training);
        }

        private Rectangle rectangularGrid() {
            Insets insets = getInsets();
            Dimension dimension = getSize();
            int x = insets.left;
            int y = insets.top;
            int width = dimension.width - insets.left - insets.right - 1;
            int height = dimension.height - insets.top - insets.bottom;
            return new Rectangle(x, y, width, height);
        }

        private void drawTraining(Graphics2D g2d, Rectangle grid) {
            for (int x = 0; x < grid.width; x++) {
                for (int y = 0; y < grid.height; y++) {
                    double[] inputs = {1.0 * x / grid.width, 1.0 * y / grid.height};
                    double[] outputs = window.nn.feedForward(inputs);
                    ClassType classType = ClassType.determineFor(outputs);
                    g2d.setColor(classType.getColor());
                    g2d.fillRect(x, y, 2, 2);
                }
            }
        }

        private void drawControlPoints(Graphics2D g2d, Rectangle grid, boolean training) {
            g2d.setStroke(new BasicStroke(2));
            for (Sample sample : window.samples) {
                double[] inputs = sample.getInputs();
                int x = (int) inputs[0] - grid.x;
                int y = (int) inputs[1] - grid.y;
                int width = 8;
                int height = 8;
                ClassType classType = sample.getClassType();
                Shape shape = classType.createShape(new Rectangle(
                        x - width / 2, y - height / 2, width, height));
                if (training) {
                    g2d.setColor(Color.WHITE);
                    g2d.draw(shape);
                    double[] outputs = window.nn.feedForward(
                            new double[]{window.transformX(inputs[0]), window.transformY(inputs[1])});
                    ClassType actualClass = ClassType.determineFor(outputs);
                    if (!classType.equals(actualClass)) {
                        g2d.setColor(Color.BLACK);
                    }
                } else {
                    g2d.setColor(classType.getColor());
                }
                g2d.fill(shape);
            }
        }

    }

}
