package hr.fer.zemris.fthesis.gui;

import hr.fer.zemris.fthesis.ann.NeuralNetwork;
import hr.fer.zemris.fthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.fthesis.ann.afunction.LReLU;
import hr.fer.zemris.fthesis.ann.afunction.Sigmoid;
import hr.fer.zemris.fthesis.ann.afunction.Tanh;
import hr.fer.zemris.fthesis.ann.dataset.Cartesian2DDataset;
import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.fthesis.ann.dataset.model.Sample;
import hr.fer.zemris.fthesis.ann.dataset.model.classes.ClassA;
import hr.fer.zemris.fthesis.ann.dataset.model.classes.ClassB;
import hr.fer.zemris.fthesis.ann.dataset.model.classes.ClassC;
import hr.fer.zemris.fthesis.ann.dataset.model.classes.ClassType;
import hr.fer.zemris.fthesis.util.Rectangle2D;

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

    // ----HELPER VARIABLES---- //
    private static final List<Sample> samples = new ArrayList<>();
    private final List<ActivationFunction> functions = List.of(
            new Sigmoid(), new LReLU(0.2), new Tanh());
    private static boolean training;
    private static volatile boolean clear;
    private static NeuralNetwork nn;
    // ------------------------ //

    // ----CANVAS COMPONENT---- //
    private static final int CANVAS_WIDTH = 400;
    private static final int CANVAS_HEIGHT = 300;
    private JComponent canvas;
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

    public Window() {
        setResizable(false);
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
        fldHiddenLayers = new JTextField("5,5", 10);
        panelHiddenLayers.add(lblHiddenLayers);
        panelHiddenLayers.add(fldHiddenLayers);
        panelOptions.add(panelHiddenLayers);
    }

    private void addSecondRow() {
        JPanel panelParams = new JPanel();
        lblIter = new JLabel("IterLimit:");
        fldIter = new JTextField("100000", 7);
        lblEta = new JLabel("Eta:");
        fldEta = new JTextField("0.1", 5);
        lblErr = new JLabel("MaxError:");
        fldErr = new JTextField("0.002", 5);
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

    private void addFourthRow() {
        JPanel panelButtons = new JPanel();
        btnGroup = new ButtonGroup();
        btnSigmoid = new JRadioButton("Sigmoid neuron");
        btnSigmoid.setSelected(true);
        btnReLu = new JRadioButton("LReLu neuron");
        btnTanh = new JRadioButton("Tanh neuron");
        btnGroup.add(btnSigmoid);
        btnGroup.add(btnReLu);
        btnGroup.add(btnTanh);
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
            clear = true;
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
            Enumeration<AbstractButton> buttons = btnGroup.getElements();
            ActivationFunction activationFunction = null;
            while (buttons.hasMoreElements()) {
                AbstractButton btn = buttons.nextElement();
                if (btn.isSelected()) {
                    switch (btn.getText().split("\\s+")[0]) {
                        case "Sigmoid":
                            activationFunction = functions.get(0);
                            break;
                        case "LReLu":
                            activationFunction = functions.get(1);
                            break;
                        case "Tanh":
                            activationFunction = functions.get(2);
                            break;
                        default:
                    }
                    break;
                }
            }
            List<Sample> normalized = new ArrayList<>();
            for (Sample s : samples) {
                double[] sInputs = s.getInputs();
                double[] inputs = {transformX(sInputs[0]), transformY(sInputs[1])};
                normalized.add(new Sample(inputs, s.getOutputs()));
            }
            ReadOnlyDataset dataset = new Cartesian2DDataset(normalized);
            nn = new NeuralNetwork(layers, activationFunction, dataset);
            nn.setCanvas(canvas);
            nn.setRedrawEveryNIter(1000);
            int iterLimit = Integer.parseInt(fldIter.getText());
            double maxError = Double.parseDouble(fldErr.getText());
            double eta = Double.parseDouble(fldEta.getText());
            btnStop.setEnabled(true);
            new Thread(() -> {
                training = true;
                nn.train(iterLimit, maxError, eta);
            }).start();
        });
        btnStop.addActionListener(evt -> {
            nn.stop();
            System.out.println("Stopped");
        });
    }
    // ------------------------------------- //

    // ---- INITIALIZE CANVAS COMPONENT ! ----- //
    private void initCanvasComponent() {
        canvas = new CanvasComponent(CANVAS_WIDTH, CANVAS_HEIGHT);
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
                    double[] outputs = currentClassType.getOutputs();
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
        public CanvasComponent(int width, int height) {
            setPreferredSize(new Dimension(width, height));
            setBorder(new TitledBorder(new EtchedBorder(), "Canvas"));
        }

        @Override
        protected void paintComponent(Graphics g) {
            Rectangle2D grid = rectangularGrid();
            Graphics2D g2d = (Graphics2D) g.create(grid.x, grid.y, grid.width, grid.height);

            // clear background
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, grid.width, grid.height);

            boolean training = Window.training;
            if (training) {
                drawTraining(g2d, grid);
            }
            drawControlPoints(g2d, grid, training);
            //Window.training = false;
        }

        private void drawControlPoints(Graphics2D g2d, Rectangle2D grid, boolean training) {
            for (Sample sample : samples) {
                double[] inputs = sample.getInputs();
                int x = (int) inputs[0] - grid.x;
                int y = (int) inputs[1] - grid.y;
                int width = 5;
                int height = 5;
                ClassType classType = sample.getClassType();
                Shape shape = classType.createShape(new Rectangle2D(x - width / 2, y - height / 2, width,
                        height));
                if (training) {
                    g2d.setColor(Color.BLACK);
                    g2d.draw(shape);
                } else {
                    g2d.setColor(classType.getColor());
                    g2d.fill(shape);
                }
            }
        }

        private void drawTraining(Graphics2D g2d, Rectangle2D grid) {
            for (int x = 0; x < grid.width; x++) {
                for (int y = 0; y < grid.height; y++) {
                    double[] inputs = {1.0 * x / grid.width, 1.0 * y / grid.height};
                    double[] outputs = Arrays.stream(nn.feedForward(inputs)).map(Math::abs).toArray();
                    ClassType classType = ClassType.forOutputs(outputs);
                    g2d.setColor(classType.getColor());
                    g2d.fillRect(x, y, 2, 2);
                }
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
