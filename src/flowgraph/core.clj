(ns flowgraph.core
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]))

(import '[java.util Random])

(m/set-current-implementation :vectorz)  ;; use Vectorz as default matrix


(defn sample-float [a b]
   (+ a (* (rand) (- b a))))

(defn sample-gaussian [rng mu std]
   (+ mu (* (.nextGaussian rng)  std)))

(defn getRandomValue [a b]
  (sample-float a b))


(defn getRandomVector [ncol std]
  (m/array (repeatedly ncol #(sample-float (* -1 std) std))))

(defn getConstantVector [ncol n]
  (m/array (repeatedly ncol (fn [] n))))

(defn getConstantMatrix [nrow ncol n]
  (m/matrix (repeatedly nrow (fn [] (getConstantVector ncol n)))))

(defn getRandomMatrix [nrow ncol std]
  (let [rng (Random.)]
    (m/matrix
      (repeatedly
        nrow
        (fn []
          (getRandomVector ncol std))))))

(defn tanh [m]
  (m/emap (fn [x] (Math/tanh x)) m))

(defn tanh_d [m] ;grad for z = tanh(x) is (1 - z^2)
 (m/emap (fn [x] (- 1.0 (* x x))) m))

(defn sigmoid [t]
  (/ (+ 1 (Math/exp (- t )))))

(defn sigmoid_d [t]
  (* t (- 1 t )))



(defn toGraphObject [x]
  (cond
    (keyword? x)
    {:forward (fn [inputs]
                (assert (contains? inputs x))
                {:output (x inputs)})
     :backward (fn [inputs] inputs)}
    (or (m/matrix? x) (m/vec? x) (number? x))
    {:forward (fn [inputs] {:output x})
     :backward (fn [inputs] inputs)}
    (= (:type x) :input)
    {:forward (fn [inputs]
                (let [k (:name x)]
                  (assert (contains? inputs k))
                  {:output (k inputs)}
                  ))
     :backward (fn [inputs]
                 inputs)}
    (= (:type x) :variable)
    {:forward (fn [inputs]
                (let [k (:name x)]
                  (assert (contains? inputs k))
                  {:output (k inputs)}))
     :backward (fn [inputs]
                (let [k (:name x)
                      value (:value (k inputs))
                      grad (:grad inputs)
                      learning_rate (:learning_rate (k inputs))
                      lgrad (m/mul learning_rate grad)
                      odelta (m/mmul lgrad value )
                      result (m/add value odelta)]
                 (assoc inputs k (assoc (k inputs) :value result))))}
    :else
    x
  ))

(defn sub [& args]
 (let [args     (map toGraphObject args)
       g_type   {:type  :mul}
       ff       (fn [inputs]
                   (let [os (map (fn [x] ((:forward x) inputs)) args)]
                     (assoc g_type
                            :inputs inputs
                            :output (reduce - (map (fn [x] (:output x)) os)))))]
   {:forward  ff
    :backward (fn [inputs]
                 (map (fn [x] ((:backward x) inputs)) args))}))

(defn sum [& args]
 (let [args     (map toGraphObject args)
       g_type   {:type  :mul}
       ff       (fn [inputs]
                   (let [os (map (fn [x] ((:forward x) inputs)) args)]
                     (assoc g_type
                            :inputs inputs
                            :output (reduce + (map (fn [x] (:output x)) os)))))]
   {:forward  ff
    :backward (fn [inputs]
                 (map (fn [x] ((:backward x) inputs)) args))}))


(defn mul [& args]
 (let [args     (map toGraphObject args)
       g_type   {:type  :mul}
       ff       (fn [inputs]
                   (let [os (map (fn [x] ((:forward x) inputs)) args)]
                     (assoc g_type
                            :inputs inputs
                            :output (reduce * (map (fn [x] (:output x)) os)))))]
   {:forward  ff
    :backward (fn [grad inputs]
                  (map (fn [x] (comment (:backward x) inputs)
                         grad) (filter #(= (:type %) :variable) args)))}))

(defn div [x y]
 (let [xg     (toGraphObject x)
       yg     (toGraphObject y)
       g_type   {:type  :div}
       ff       (fn [inputs]
                   (let [xo ((:forward xg) inputs)
                         yo ((:forward yg) inputs)]
                     (assoc g_type
                            :inputs inputs
                            :output (/ (:output xo) (:output yo)))))]
   {:forward  ff
    :backward (fn [inputs]
                 ((:backward xg) inputs))}))

(defn mmul [w v]
 (let [wi (toGraphObject w)
       vi (toGraphObject v)
       ob {:type            :mmul
           :adjusted_output []}
       ff (fn [inputs]
               (let [wo ((:forward wi) inputs)
                     vo ((:forward vi) inputs)]
                 (assoc ob
                        :inputs [wo vo]
                        :output (m/mmul (:output vo) (:output wo)))))]
   {:forward  ff
    :backward (fn [inputs]
               ((:backward vi) ((:backward wi) inputs )))}))



(defn sqr [v]
 (let [vi (toGraphObject v)
       g_type {:type            :sqr}
       ff (fn [inputs]
               (let [vo ((:forward vi) inputs)]
                 (assoc g_type
                        :inputs vo
                        :output (:output vo))))]
   {:forward  ff
    :backward (fn [inputs]
               ((:backward vi) inputs))}))

(defn minimize [g]
  g)

(defn input [n]
  {:name n :type :input})

(defn variable [n]
  {:name n :type :variable})

(defn graph [t]
  t)


(defn predict [mgraph variables data]
  ((:forward mgraph) (merge (map variable variables) (map input data))))

(defn solve [mgraph evaluator variables data]
  (let [pred   (predict mgraph variables data)
        inputs (merge (map variable variables) (merge (:output pred)   (map input data)))
        ev     ((:forward evaluator) inputs)]
    ((:backward mgraph) (:output evaluator) inputs)))



(defn iter [mgraph evaluator variables data]
  (loop [idx       0
         variables variables
         o         nil]
         (if (>= idx (count data))
            {:variables variables :output o}
            (recur (inc idx)
                   (solve   mgraph evaluator variables (nth data idx))
                   (predict mgraph variables (nth data idx))))))



