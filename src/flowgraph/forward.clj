(ns flowgraph.forward
  (:require
    [flowgraph.core :as fg]
    [clojure.core.matrix.random :as r]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]
    [clojure.core.async :as a :refer [>! <! >!! <!! go go-loop chan buffer close! thread alts! alts!! timeout]]
    )
   (:require [incanter core charts])
   (:import [org.jfree.chart ChartPanel JFreeChart])
   (:import [java.awt.event ActionEvent ActionListener])
   (:import [javax.swing JComponent JLabel JPanel])
   (:import [java.util Random]))




(m/set-current-implementation :vectorz)  ;; use Vectorz as default matrix

(defn sigmoid-activation [weight bias input]
  (m/logistic (m/add (m/mmul weight input) bias)))

(defn sigmoid-back [output target weight bias learning_rate]
  (let [diff   (m/mul (m/sub  target output) 2)
        deriv  (m/mul output (m/sub 1.0 output))
        gderiv (m/mul diff deriv)
        g_w    (m/mmul (m/transpose weight) gderiv)
        new_w  (m/add weight (m/mul learning_rate g_w))
        new_b  (m/add bias   (m/mul learning_rate gderiv))
        sumdif (reduce + diff)]
    {:weight new_w
     :bias new_b
     :difference sumdif}))

(defn sigmoid-layer [options input-chan]
  (let [out-chan   (chan)
        maint-chan (chan)]
    (go-loop [w (:weight        options)
              b (:bias          options)
              l (:learning_rate options)
              i (<! input-chan)]
               (let [output (sigmoid-activation w b i)]
                 (>! out-chan output)
                 (let [ni (<! input-chan)
                       nw (sigmoid-back output ni w b l)]
                   (>! maint-chan (assoc nw :id (:id options)))
               (recur (:weight nw)
                      (:bias nw)
                      l
                      ni))))
    {:out   out-chan
     :maint maint-chan}))

(defn createLayer [values input-chan & {:keys [learning_rate] :or {learning_rate 0.1}}]
  (let [l   (sigmoid-layer (assoc values  :learning_rate learning_rate) input-chan)]
    l))

(defn createStack [items input-chan]
  (loop [idx 0
         prev-chan input-chan
         package []]
    (if (>= idx (count items))
      {:out prev-chan :maints package}
      (let [item  (nth items idx)
            l     (createLayer item prev-chan)
            out   (:out l)
            maint (:maint l)]
      (recur (inc idx) out (conj package maint))))))


(defn stack [options input-chan]
  (let [id (:id options)
        items (:stacks options)
        out-chan   (chan)
        maint-chan (chan)
        s          (createStack items input-chan)]
        ;m          (a/merge (:maints s))]
    (go-loop []
         (>! out-chan (<! (:out s)))
         (doseq [i (:maints s)]
           (>! maint-chan (<! i)))
        (recur))
  {:out out-chan
   :maint maint-chan}))

(defn getNormalVector[size]
     (r/sample-normal size))

(defn getNormalMatrix [size]
     (m/array (repeatedly size #(r/sample-normal size))))

(defn init-Layer [id size]
  (let [initial-weights (getNormalMatrix size)
        initial-bias  (getNormalVector size)]
    {:id id :weight initial-weights :bias initial-bias}))


(defn mapNum2Vec [n]
  (map (fn [x]
         (if  (= n x) 1 0)) (range 9)))

(defn mapVec2Num [v]
  (reduce + (map-indexed (fn [idx x]
         (if (zero? x) 0 idx)) v)))





(defn print-progress [c]
  (go (while true
        (let [i (<! c)]
        (println (:id i) " = " (:difference i))))))



(defn default-dimensions
  (^java.awt.Dimension []
    (java.awt.Dimension. 400 300)))

(defn time-chart
  ([input-chan
    & {:keys [repaint-speed time-periods y-min y-max]
       :or {repaint-speed 250
            time-periods 1200}}]
    (let [start-millis (System/currentTimeMillis)
          times (atom  [])
          values (atom [])
          next-chart  (fn []
                         (let [chart (incanter.charts/time-series-plot @times @values :x-label "time" :y-label "values")]
                           (if y-max (incanter.charts/set-y-range chart (double (or y-min 0.0)) (double y-max)))
                           chart))
          panel (ChartPanel. ^JFreeChart (next-chart))
          timer (javax.swing.Timer.
                  (int repaint-speed)
                  (proxy [java.awt.event.ActionListener] []
                    (actionPerformed
                      [^ActionEvent e]
                      (when (.isShowing panel)
                        (.setChart panel ^JFreeChart (next-chart))
                        (.repaint ^JComponent panel)) )))]
      (go-loop []
        (let [i (<! input-chan)]
         (swap! values conj i)
         (swap! times  conj (System/currentTimeMillis))
         (recur)))
      (.start timer)
      (.setPreferredSize panel (default-dimensions))
      panel)))



(defn show [item]
  (let [f (javax.swing.JFrame. "plot")]
      (.add (.getContentPane f) item)
      (.setMinimumSize f (default-dimensions))
      (.setVisible f true)))

(defn print-simple-progress [c]
  (go (while true
        (let [i (<! c)]
        (println i)))))



(defn generate-m2 []
  (flatten (map (fn [x] [(m/array (mapNum2Vec x)) (m/array (mapNum2Vec (* x 2)))])
                            (flatten (repeatedly 1000 #(range 5))))))



(defn sample-random-timed-data [amount]
  (let [out-chan (chan)]
    (go-loop [ind 0]
             (if (>= ind amount)
               nil
               (let [t (<! (timeout 300))]
                 (>! out-chan (rand-int 20))
                 (recur (inc ind)))))
    out-chan))

(defn test-l []
 (let [data (generate-m2)
       input-chan (chan)
       l  (createLayer (init-Layer "test" 9) input-chan)]
   ;(print-progress (:maint l))
   (show (time-chart (:maint l)))
   (go
       (doseq [i data]
         (>!! input-chan i)
         (<! (:out l))))))

(defn test-m []
 (let [data (generate-m2)
       input-chan (chan)
       l  (stack {:id "stack-test" :stacks [(init-Layer "l1" 9) (init-Layer "l2" 9)]} input-chan)]
   (print-progress (:maint l))
   (go
       (doseq [i data]
         (>!! input-chan i)
         (<! (:out l))))))




