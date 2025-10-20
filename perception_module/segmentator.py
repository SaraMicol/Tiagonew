import cv2
import numpy as np

def check_white_inside(image, bbox, threshold=0.3, simulated=False):
    """
    Verifica se c'� abbastanza bianco (o marrone se simulato) dentro il rettangolo
    threshold: percentuale minima di pixel richiesta (0.0-1.0)
    simulated: se True, cerca marrone del tavolo invece che bianco
    """
    x, y, w, h = bbox
    
    # Estrai la regione del rettangolo
    roi = image[y:y+h, x:x+w]
    
    # Converti in HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    if simulated:
        # Maschera per il marrone del tavolo in Gazebo
        lower_brown = np.array([10, 50, 50])   # Range marrone/arancione
        upper_brown = np.array([30, 255, 200])
        target_mask = cv2.inRange(hsv_roi, lower_brown, upper_brown)
    else:
        # Maschera per il bianco (bassa saturazione, alta luminosit�)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        target_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
    
    # Calcola la percentuale di pixel target
    target_pixels = cv2.countNonZero(target_mask)
    total_pixels = w * h
    target_percentage = target_pixels / total_pixels if total_pixels > 0 else 0
    
    return target_percentage >= threshold, target_percentage

def find_line_rectangles(image, color='blue', debug=False, simulated=False):
    """
    Trova rettangoli formati da LINEE colorate (non aree piene)
    con BIANCO (o MARRONE se simulato) all'interno
    simulated: se True, cerca marrone del tavolo invece di bianco
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Range di colore
    if color == 'blue':
        if simulated:
            # Range pi� ampio per Gazebo
            lower_bound = np.array([90, 50, 50])
            upper_bound = np.array([135, 255, 255])
        else:
            lower_bound = np.array([100, 100, 50])
            upper_bound = np.array([130, 255, 255])
        draw_color = (255, 0, 0)
    elif color == 'green':
        if simulated:
            # Range pi� ampio per Gazebo
            lower_bound = np.array([30, 50, 50])
            upper_bound = np.array([90, 255, 255])
        else:
            lower_bound = np.array([35, 100, 50])
            upper_bound = np.array([85, 255, 255])
        draw_color = (0, 255, 0)
    else:
        return []
    
    # Crea maschera
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    if debug:
        cv2.imshow(f"Maschera {color}", mask)
    
    # Dilata leggermente per collegare linee spezzate
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Trova contorni
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    
    for contour in contours:
        # Calcola bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtra per dimensione minima
        min_size = 30 if simulated else 50
        if w > min_size and h > min_size:
            # Approssima con poligono
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Cerca forme con 4-8 vertici (rettangoli possono avere vertici extra)
            max_vertices = 12 if simulated else 8
            if 4 <= len(approx) <= max_vertices:
                # VERIFICA SE C'� BIANCO/MARRONE DENTRO
                has_target, target_pct = check_white_inside(image, (x, y, w, h), 
                                                            threshold=0.3, 
                                                            simulated=simulated)
                
                if has_target:
                    area = cv2.contourArea(contour)
                    
                    # Calcola aspect ratio del bounding box
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Calcola quanto il contorno riempie il bounding box
                    extent = area / (w * h) if (w * h) > 0 else 0
                    
                    rectangles.append({
                        'contour': approx,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'white_percentage': target_pct,
                        'color': color,
                        'draw_color': draw_color
                    })
    
    return rectangles

def get_rectangle_corners(bbox):
    """
    Restituisce i 4 angoli del rettangolo dal bounding box
    """
    x, y, w, h = bbox
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)
    return corners

def get_rectangle_centroid(bbox):
    """
    Calcola il centroide (centro) del rettangolo
    Restituisce (cx, cy)
    """
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)

def find_all_line_rectangles(image, debug=False, simulated=False):
    """
    Trova tutti i rettangoli formati da linee blu e verdi
    simulated: se True, cerca marrone invece di bianco all'interno
    """
    blue_rectangles = find_line_rectangles(image.copy(), 'blue', debug, simulated)
    green_rectangles = find_line_rectangles(image.copy(), 'green', debug, simulated)
    
    return blue_rectangles, green_rectangles

def draw_found_rectangles(image, rectangles_list):
    """
    Disegna i rettangoli trovati con centroide
    """
    result = image.copy()
    
    for rectangles in rectangles_list:
        for i, rect in enumerate(rectangles):
            # Disegna il bounding box
            x, y, w, h = rect['bbox']
            corners = get_rectangle_corners(rect['bbox'])
            cv2.polylines(result, [corners], True, rect['draw_color'], 3)
            
            # Calcola e disegna il centroide
            centroid = get_rectangle_centroid(rect['bbox'])
            cv2.circle(result, centroid, 8, rect['draw_color'], -1)
            cv2.circle(result, centroid, 10, (0, 0, 0), 2)
            
            # Aggiungi informazioni
            label = f"{rect['color']} #{i+1}"
            info = f"{w}x{h}px"
            white_info = f"contenuto: {rect['white_percentage']*100:.0f}%"
            centroid_info = f"C: {centroid}"
            
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect['draw_color'], 2)
            cv2.putText(result, info, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect['draw_color'], 1)
            cv2.putText(result, white_info, (x, y + h + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect['draw_color'], 1)
            cv2.putText(result, centroid_info, (x, y + h + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, rect['draw_color'], 1)
    
    return result

def select_best_rectangles(blue_rectangles, green_rectangles):
    """
    Seleziona i rettangoli migliori (pi� grandi) e restituisce anche i centroidi
    """
    best_blue = None
    best_green = None
    blue_centroid = None
    green_centroid = None
    
    if blue_rectangles:
        blue_rectangles.sort(key=lambda x: x['area'], reverse=True)
        best_blue = blue_rectangles[0]
        blue_centroid = get_rectangle_centroid(best_blue['bbox'])
    if green_rectangles:
        green_rectangles.sort(key=lambda x: x['area'], reverse=True)
        best_green = green_rectangles[0]
        green_centroid = get_rectangle_centroid(best_green['bbox'])
    
    return (best_blue, blue_centroid), (best_green, green_centroid)

def obtain_centroids(image, simulated=False):
    """
    Funzione principale per ottenere i centroidi dei rettangoli blu e verdi
    simulated: se True, cerca marrone del tavolo invece di bianco all'interno
    """
    # Trova tutti i rettangoli (debug=True mostra le maschere)
    blue_rects, green_rects = find_all_line_rectangles(image, debug=False, simulated=simulated)
    
    # Seleziona i migliori e ottieni i centroidi
    (best_blue, blue_centroid), (best_green, green_centroid) = select_best_rectangles(blue_rects, green_rects)
    
    # Disegna risultati
    result = draw_found_rectangles(image, [blue_rects, green_rects])
    
    # Restituisci i centroidi per uso successivo
    return blue_centroid, green_centroid