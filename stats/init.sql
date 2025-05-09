-- 통계 데이터베이스 초기화 스크립트

-- API 호출 통계 테이블
CREATE TABLE IF NOT EXISTS api_calls (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL COMMENT 'API 엔드포인트',
    method VARCHAR(10) NOT NULL COMMENT 'HTTP 메서드',
    status_code INT NOT NULL COMMENT 'HTTP 상태 코드',
    response_time FLOAT NOT NULL COMMENT '응답 시간(초)',
    request_size INT COMMENT '요청 크기(바이트)',
    response_size INT COMMENT '응답 크기(바이트)',
    user_agent VARCHAR(255) COMMENT '사용자 에이전트',
    ip_address VARCHAR(45) COMMENT 'IP 주소',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
    INDEX idx_endpoint (endpoint),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 일별 통계 집계 테이블
CREATE TABLE IF NOT EXISTS daily_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL COMMENT 'API 엔드포인트',
    date DATE NOT NULL COMMENT '날짜',
    total_calls INT NOT NULL DEFAULT 0 COMMENT '총 호출 수',
    success_calls INT NOT NULL DEFAULT 0 COMMENT '성공 호출 수',
    error_calls INT NOT NULL DEFAULT 0 COMMENT '오류 호출 수',
    avg_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '평균 응답 시간(초)',
    max_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '최대 응답 시간(초)',
    min_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '최소 응답 시간(초)',
    total_request_size BIGINT NOT NULL DEFAULT 0 COMMENT '총 요청 크기(바이트)',
    total_response_size BIGINT NOT NULL DEFAULT 0 COMMENT '총 응답 크기(바이트)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 시간',
    UNIQUE KEY idx_endpoint_date (endpoint, date),
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 일별 통계 집계를 위한 이벤트 스케줄러
DELIMITER //
CREATE EVENT IF NOT EXISTS aggregate_daily_stats
    ON SCHEDULE EVERY 1 DAY
    STARTS CURRENT_DATE + INTERVAL 1 DAY
    DO
    BEGIN
        -- 어제 날짜의 통계 집계
        INSERT INTO daily_stats (endpoint, date, total_calls, success_calls, error_calls, 
                              avg_response_time, max_response_time, min_response_time,
                              total_request_size, total_response_size)
        SELECT 
            endpoint,
            DATE(created_at) as call_date,
            COUNT(*) as total_calls,
            SUM(CASE WHEN status_code < 400 THEN 1 ELSE 0 END) as success_calls,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_calls,
            AVG(response_time) as avg_response_time,
            MAX(response_time) as max_response_time,
            MIN(response_time) as min_response_time,
            SUM(request_size) as total_request_size,
            SUM(response_size) as total_response_size
        FROM api_calls
        WHERE DATE(created_at) = DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY)
        GROUP BY endpoint, DATE(created_at)
        ON DUPLICATE KEY UPDATE
            total_calls = VALUES(total_calls),
            success_calls = VALUES(success_calls),
            error_calls = VALUES(error_calls),
            avg_response_time = VALUES(avg_response_time),
            max_response_time = VALUES(max_response_time),
            min_response_time = VALUES(min_response_time),
            total_request_size = VALUES(total_request_size),
            total_response_size = VALUES(total_response_size),
            updated_at = CURRENT_TIMESTAMP;
    END //

-- 오래된 통계 데이터 정리를 위한 이벤트 스케줄러
CREATE EVENT IF NOT EXISTS cleanup_old_api_calls
    ON SCHEDULE EVERY 1 WEEK
    STARTS CURRENT_DATE + INTERVAL 1 WEEK
    DO
    BEGIN
        -- 30일 이상 지난 상세 로그 삭제
        DELETE FROM api_calls 
        WHERE created_at < DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY);
    END //
DELIMITER ; 