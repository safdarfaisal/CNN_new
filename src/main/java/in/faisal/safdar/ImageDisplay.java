package in.faisal.safdar;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

public class ImageDisplay {
    public static List<Optional<Image>> sampleIdsToImages(List<SampleId> sampleList, MNISTDataset ds) {
        return sampleList.stream().map(s -> {
            Optional<Image> im = Optional.empty();
            try {
                im = Optional.ofNullable(ds.trainingSample(s).bufferedImage());
            } catch(IOException ex) {
                ex.printStackTrace();
            }
            return im;
        }).toList();
    }
    public static void show(List<SampleId> sampleList, MNISTDataset ds, String text) {
        Optional<Image> o = gridImages(sampleIdsToImages(sampleList, ds), 20, Color.white, 5, text);
        o.ifPresent(ImageDisplay::show);
    }

    public static void show(Image image) {
        // create the GUI for viewing the image if needed
        JFrame frame = new JFrame();
        //set image
        frame.setContentPane(new JLabel(new ImageIcon(image)));
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setResizable(false);
        frame.pack();
        // draw
        frame.setVisible(true);
        frame.repaint();
    }

    /**
     * Merge images
     *
     * @param images list of images
     * @param space space between images
     * @param bg background color
     * @param columns number of columns
     *
     * @return merged image
     *
     * Thanks to <a href="http://www.java2s.com/example/java-utility-method/bufferedimage-merge/mergeimages-list-images-int-space-color-bg-1d30c.html">Java2S utility examples</a>
     */
    public static Optional<Image> gridImages(List<Optional<Image>> images, int space, Color bg, int columns, String text) {
        if (images.isEmpty()) {
            return Optional.empty();
        }
        if (images.size() == 1) {
            return images.get(0);
        }

        int maxHeight = 0;
        int maxWidth = 0;
        int rows = (int) (images.size() / (double) columns);

        if (rows == 0) {
            rows = 1;
        }

        for (Optional<Image> o : images) {
            if (o.isPresent()) {
                Image im = o.get();
                int imageWidth = im.getWidth(null);
                int imageHeight = im.getHeight(null);

                maxHeight = Math.max(maxHeight, imageHeight);
                maxWidth = Math.max(maxWidth, imageWidth);
            }
        }

        if (columns > images.size()) {
            columns = images.size();
        }

        BufferedImage bImage = new BufferedImage(maxWidth * columns + (columns - 1) * space,
                maxHeight * rows + (rows - 1) * space + 60, BufferedImage.TYPE_INT_RGB);
        Graphics g = bImage.getGraphics();

        if (bg != null) {
            g.setColor(bg);
            g.fillRect(0, 0, bImage.getWidth(null), bImage.getHeight(null));
        }

        int colCnt = 0;
        int rowCnt = 0;

        for (Optional<Image> o : images) {
            if (o.isPresent()) {
                Image im = o.get();
                g.drawImage(im, colCnt * (maxWidth + space), rowCnt * (maxHeight + space), null);
                colCnt++;

                if (colCnt >= columns) {
                    colCnt = 0;
                    rowCnt++;
                }
            }
        }

        Font font = new Font("Courier New", Font.PLAIN, 12);
        g.setFont(font);
        g.setColor(Color.black);
        g.drawString(text, 5, bImage.getHeight()-30);

        return Optional.of(bImage);
    }
}
