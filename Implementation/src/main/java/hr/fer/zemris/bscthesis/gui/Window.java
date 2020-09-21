package hr.fer.zemris.bscthesis.gui;

import hr.fer.zemris.bscthesis.ann.NeuralNetwork;
import hr.fer.zemris.bscthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.bscthesis.ann.afunction.ReLU;
import hr.fer.zemris.bscthesis.ann.afunction.Sigmoid;
import hr.fer.zemris.bscthesis.ann.afunction.Tanh;
import hr.fer.zemris.bscthesis.ann.dataset.Cartesian2DDataset;
import hr.fer.zemris.bscthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.bscthesis.ann.dataset.model.Sample;
import hr.fer.zemris.bscthesis.classes.ClassA;
import hr.fer.zemris.bscthesis.classes.ClassB;
import hr.fer.zemris.bscthesis.classes.ClassC;
import hr.fer.zemris.bscthesis.classes.ClassType;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

public class Window extends JFrame {

    // ----NEURAL NETWORK STUFF---- //
    private final NeuralNetwork nn = new NeuralNetwork();
    private NeuralNetwork.LearningType learningType;
    private int batchSize;
    private ActivationFunction aFunction;
    private final ReadOnlyDataset dataset = new Cartesian2DDataset();
    private final List<Sample> samples = new ArrayList<>();
    private final List<Sample> normalized = new ArrayList<>();
    private final List<ActivationFunction> aFunctions = List.of(
            new Sigmoid(), new ReLU(), new Tanh());
    // ---------------------------- //

    // ----CANVAS COMPONENT---- //
    private static final int CANVAS_WIDTH = 400;
    private static final int CANVAS_HEIGHT = 300;
    private JComponent canvas;
    // ------------------------ //

    // ----OPTIONS PANEL---- //
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
    private JRadioButton btnSigmoid;
    private JRadioButton btnReLu;
    private JRadioButton btnTanh;
    private JButton btnTrain;
    private JButton btnStop;
    private JButton btnDeleteLast;
    private JButton btnClearAll;
    // -------------------- //

    // --- CLASS TYPE STUFF --- //
    private final List<ClassType> classTypes = List.of(new ClassA(), new ClassB(), new ClassC());
    private ClassType currentClassType = classTypes.get(0);
    // ------------------------ //

    // --- HELPER VARIABLES --- //
    private boolean training;
    // ------------------------ //

    // ---- CONSTRUCTOR ---- //
    public Window() {
        setResizable(false);
        setTitle("ANN classification");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        initGUI();
        pack();
        setLocationRelativeTo(null);
    }
    // --------------------- //

    private void initGUI() {
        initPanelOptions();
        initCanvasComponent();
    }

    // ---- INITIALIZE OPTIONS PANEL ! ----- //
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
        boxBPType = new JComboBox<>(new String[]{"Stochastic", "Batch", "Mini_Batch"});
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
        fldClassType.setText(currentClassType.toString());
        panelClassType.add(lblClassType);
        panelClassType.add(fldClassType);
        panelOptions.add(panelClassType);
    }

    private void addFifthRow() {
        JPanel panelButtons = new JPanel();
        btnGroup = new ButtonGroup();
        btnSigmoid = new JRadioButton("Sigmoid neuron");
        btnSigmoid.setSelected(true);
        btnReLu = new JRadioButton("ReLu neuron");
        btnTanh = new JRadioButton("Tanh neuron");
        btnGroup.add(btnSigmoid);
        btnGroup.add(btnReLu);
        btnGroup.add(btnTanh);
        panelButtons.add(btnSigmoid);
        panelButtons.add(btnReLu);
        panelButtons.add(btnTanh);
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
            Enumeration<AbstractButton> buttons = btnGroup.getElements();
            while (buttons.hasMoreElements()) {
                AbstractButton btn = buttons.nextElement();
                if (btn.isSelected()) {
                    switch (btn.getText().split("\\s+")[0]) {
                        case "Sigmoid":
                            aFunction = aFunctions.get(0);
                            break;
                        case "ReLu":
                            aFunction = aFunctions.get(1);
                            break;
                        case "Tanh":
                            aFunction = aFunctions.get(2);
                            break;
                        default:
                    }
                    break;
                }
            }
            nn.setAFunction(aFunction);
            normalized.clear();
            for (Sample s : samples) {
                double[] sInputs = s.getInputs();
                double[] inputs = {transformX(sInputs[0]), transformY(sInputs[1])};
                normalized.add(new Sample(inputs, s.getOutputs()));
            }
            dataset.setSamples(normalized);
            nn.setDataset(dataset);
            String type = (String) boxBPType.getSelectedItem();
            if (type.equals("Stochastic")) {
                learningType = NeuralNetwork.LearningType.ONLINE;
            } else if (type.equals("Batch")) {
                learningType = NeuralNetwork.LearningType.BATCH;
            } else {
                learningType = NeuralNetwork.LearningType.MINI_BATCH;
            }
            if (learningType == NeuralNetwork.LearningType.MINI_BATCH) {
                do {
                    String bSize = JOptionPane.showInputDialog(this, "Enter batch size:");
                    if (bSize == null || !bSize.matches("^[1-9]\\d*$")) {
                        JOptionPane.showMessageDialog(this, "Enter natural number.");
                    } else {
                        batchSize = Integer.parseInt(bSize);
                        break;
                    }
                } while (true);
            }
            nn.setLearningType(learningType);
            nn.setBatchSize(batchSize);
            nn.setCanvas(canvas);
            nn.setRedrawEveryNEpoch(1000);
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
        });
        btnStop.addActionListener(evt -> nn.stop());
    }
    // ------------------------------------- //

    // ---- INITIALIZE CANVAS COMPONENT ! ----- //
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
                    fldClassType.setText(currentClassType.toString());
                } else {
                    double[] inputs = {e.getX(), e.getY()};
                    double[] outputs = currentClassType.getDesiredOutputs();
                    Sample sample = new Sample(inputs, outputs);
                    sample.setClassType(currentClassType);
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
    // ---------------------------------------- //

    private double transformX(double x) {
        return x / canvas.getSize().getWidth();
    }

    private double transformY(double y) {
        return 1.0 * y / canvas.getSize().getHeight();
    }

    // ---------------------------------------- //

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

        private void drawControlPoints(Graphics2D g2d, Rectangle grid, boolean training) {
            g2d.setStroke(new BasicStroke(2));
            for (Sample sample : window.samples) {
                double[] inputs = sample.getInputs();
                int x = (int) inputs[0] - grid.x;
                int y = (int) inputs[1] - grid.y;
                int width = 8;
                int height = 8;
                ClassType classType = sample.getClassType();
                Shape shape = classType.createShape(new Rectangle(x - width / 2, y - height / 2, width,
                        height));
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

        private Rectangle rectangularGrid() {
            Insets insets = getInsets();
            Dimension dimension = getSize();
            int x = insets.left;
            int y = insets.top;
            int width = dimension.width - insets.left - insets.right - 1;
            int height = dimension.height - insets.top - insets.bottom;
            return new Rectangle(x, y, width, height);
        }

    }

}
